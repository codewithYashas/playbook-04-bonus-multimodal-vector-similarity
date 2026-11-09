#!/bin/bash
# ByteEats Vector Similarity Search Pipeline Runner
# This script runs the various components of the ByteEats system

# Every Python command runs through uv, in the environment that install.sh (uv sync) creates.

# Default configuration
API_PORT=${API_PORT:-8080}
DASHBOARD_PORT=${DASHBOARD_PORT:-8050}
IMAGE_DIR=${IMAGE_DIR:-"data/images"}
VIDEO_DIR=${VIDEO_DIR:-"data/videos"}
QUALITY_DIR=${QUALITY_DIR:-"data/quality"}
EMBEDDINGS_DIR=${EMBEDDINGS_DIR:-"data/embeddings"}

# Display help
display_help() {
    echo "ByteEats Vector Similarity Search Pipeline Runner"
    echo ""
    echo "Usage: ./run_pipeline.sh [options]"
    echo ""
    echo "Options:"
    echo "  --setup                 Setup directory structure and check paths"
    echo "  --start-api             Start the similarity search API"
    echo "  --start-dashboard       Start the visualization dashboard"
    echo "  --start-quality-api     Start the quality check API"
    echo "  --quality-check         Run the quality check tool"
    echo "  --list-images           List reference images"
    echo "  --list-videos           List available videos"
    echo "  --move-images <dir>     Move images from directory to reference_images"
    echo "  --generate-embeddings   Generate embeddings from videos and images"
    echo "  --help                  Display this help message"
    echo ""
    echo "Environment variables:"
    echo "  API_PORT                Port for the API server (default: 8080)"
    echo "  DASHBOARD_PORT          Port for the dashboard server (default: 8050)"
    echo "  API_URL                 URL for the API server (default: http://localhost:<API_PORT>)"
    echo "  IMAGE_DIR               Directory for reference images (default: data/images)"
    echo "  VIDEO_DIR               Directory for videos (default: data/videos)"
    echo "  QUALITY_DIR             Directory for quality reports (default: data/quality)"
    echo "  EMBEDDINGS_DIR          Directory for embeddings (default: data/embeddings)"
    echo "  HOST                    Address the servers listen on (default: 127.0.0.1, this machine only)"
    echo ""
}

# Setup directory structure
setup_directories() {
    echo "Setting up directory structure..."
    uv run --frozen python scripts/setup_directories.py
}

# Start the similarity search API
start_api() {
    echo "Starting the similarity search API on port ${API_PORT}..."
    export PORT=${API_PORT}
    export IMAGE_DIR=${IMAGE_DIR}
    export VIDEO_DIR=${VIDEO_DIR}
    export QUALITY_DIR=${QUALITY_DIR}
    export EMBEDDINGS_DIR=${EMBEDDINGS_DIR}
    uv run --frozen python src/api/main.py
}

# Start the visualization dashboard
start_dashboard() {
    echo "Starting the visualization dashboard on port ${DASHBOARD_PORT}..."
    export DASHBOARD_PORT=${DASHBOARD_PORT}
    export API_URL=${API_URL:-"http://localhost:${API_PORT}"}
    uv run --frozen python src/api/dashboard.py
}

# Start the quality check API
start_quality_api() {
    echo "Starting the quality check API on port ${API_PORT}..."
    export PORT=${API_PORT}
    export IMAGE_DIR=${IMAGE_DIR}
    export VIDEO_DIR=${VIDEO_DIR}
    export QUALITY_DIR=${QUALITY_DIR}
    uv run --frozen python src/api/run_quality_api.py
}

# Run the quality check tool
quality_check() {
    echo "Running the quality check tool..."
    API_URL="http://localhost:${API_PORT}"
    echo "Using API URL: ${API_URL}"
    uv run --frozen python scripts/quality_check.py --api-url ${API_URL}
}

# List reference images
list_images() {
    uv run --frozen python scripts/setup_directories.py --list-images
}

# List available videos
list_videos() {
    export API_URL=${API_URL:-"http://localhost:${API_PORT}"}
    uv run --frozen python scripts/quality_check.py --list-videos
}

# Move images from directory to reference_images
move_images() {
    if [ -z "$1" ]; then
        echo "Error: No source directory specified."
        exit 1
    fi
    uv run --frozen python scripts/setup_directories.py --move-images "$1"
}

# Generate embeddings from videos and images
generate_embeddings() {
    echo "Generating embeddings from videos and images..."
    mkdir -p ${EMBEDDINGS_DIR}
    uv run --frozen python src/embeddings/generator.py --input ${VIDEO_DIR},${IMAGE_DIR} --output ${EMBEDDINGS_DIR}
    echo "Embeddings generation complete."
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    display_help
    exit 1
fi

# Process arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --setup)
            setup_directories
            shift
            ;;
        --start-api)
            start_api
            shift
            ;;
        --start-dashboard)
            start_dashboard
            shift
            ;;
        --start-quality-api)
            start_quality_api
            shift
            ;;
        --quality-check)
            quality_check
            shift
            ;;
        --list-images)
            list_images
            shift
            ;;
        --list-videos)
            list_videos
            shift
            ;;
        --move-images)
            move_images "$2"
            shift 2
            ;;
        --generate-embeddings)
            generate_embeddings
            shift
            ;;
        --help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

echo "Pipeline execution complete." 