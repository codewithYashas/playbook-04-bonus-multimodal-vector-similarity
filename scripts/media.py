"""ResNet-18 embeddings for videos, frames and images, and JPEG thumbnails.

The numbers match the original playbook's ``MediaEmbeddingGenerator``: every 5th frame
of a video goes through ResNet-18 (ImageNet weights, classifier removed, 224x224
input), the raw 512-d features are averaged and the mean is L2-normalised. Images go
through the same model. What changed is where the vectors go: into Lance tables, not
``.npy`` files.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
from PIL import Image

FRAME_SAMPLE_RATE = 5  # every 5th frame, as in the original generator
THUMB_WIDTH = 320
FRAME_THUMB_WIDTH = 224
JPEG_QUALITY = 82


def normalise(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def jpeg_bytes(image: Image.Image, width: int) -> bytes:
    """A JPEG of ``image`` scaled to ``width`` pixels wide."""
    image = image.convert("RGB")
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


@dataclass
class VideoFrames:
    frames: list[np.ndarray]  # RGB
    indices: list[int]
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def read_frames(path: str, every: int = FRAME_SAMPLE_RATE) -> VideoFrames:
    """Every ``every``-th frame of a video, RGB, seeking as the original generator did."""
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, indices = [], []
    for i in range(0, frame_count, every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            indices.append(i)
    cap.release()
    return VideoFrames(frames, indices, fps, frame_count, width, height)


class Embedder:
    """ResNet-18 feature extractor; the model loads on first use (about 45 MB, cached by torch)."""

    def __init__(self, device: str | None = None) -> None:
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def _model(self):  # noqa: ANN202
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device)
        model.eval()
        return model

    @cached_property
    def _transform(self):  # noqa: ANN202
        import torchvision.transforms as T

        return T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def features(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Raw (unnormalised) 512-d features, one row per image."""
        import torch

        out = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = torch.stack([self._transform(im.convert("RGB")) for im in images[start : start + batch_size]])
                out.append(self._model(batch.to(self.device)).flatten(1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.empty((0, 512), np.float32)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """A normalised embedding for one image."""
        return normalise(self.features([image])[0])

    def embed_frames(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Normalised per-frame embeddings and the video embedding (normalised mean of raw features)."""
        raw = self.features([Image.fromarray(f) for f in frames])
        per_frame = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        return per_frame.astype(np.float32), normalise(raw.mean(axis=0))
