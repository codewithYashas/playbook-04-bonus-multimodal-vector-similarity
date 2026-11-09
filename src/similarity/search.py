#!/usr/bin/env python3
"""
Similarity Search Module for ByteEats Vector Similarity Search

This module implements efficient similarity search for video embeddings
using LanceDB, an embedded vector database. The embeddings and their
metadata live in a Lance table on local disk, so there is no server to run.
"""

import os
import json
import time
import numpy as np
import pyarrow as pa
import lancedb
from datetime import datetime
from collections import defaultdict

TABLE_NAME = "embeddings"

# The index types this playbook compares, mapped to their LanceDB equivalents.
# All three keep full-precision vectors, so their scores match exact search.
INDEX_TYPES = {
    "Flat": None,               # No index: exact (brute-force) search
    "IVF": "IVF_FLAT",          # Inverted file index over k-means partitions
    "HNSW": "IVF_HNSW_FLAT",    # HNSW graph inside each IVF partition
}


def default_db_uri(embedding_dir):
    """LanceDB directory: $LANCEDB_URI, or data/lancedb next to the embeddings folder."""
    return os.environ.get("LANCEDB_URI") or os.path.join(
        os.path.dirname(os.path.abspath(embedding_dir)), "lancedb")


class SimilaritySearch:
    def __init__(self, embedding_dir, db_uri=None):
        """Initialize the similarity search with a directory of embeddings."""
        self.embedding_dir = embedding_dir
        self.db_uri = db_uri or default_db_uri(embedding_dir)
        self.embeddings = []
        self.metadata = []
        self.video_ids = []
        self.db = None
        self.table = None
        self.index = None
        self.index_type = None

    def load_embeddings(self):
        """Load embeddings from the specified directory into a LanceDB table."""
        embedding_files = [f for f in os.listdir(self.embedding_dir) if f.endswith('_embedding.npy')]

        print(f"Loading {len(embedding_files)} embeddings...")

        for embedding_file in embedding_files:
            video_id = embedding_file.replace('_embedding.npy', '')
            embedding_path = os.path.join(self.embedding_dir, embedding_file)
            metadata_path = os.path.join(self.embedding_dir, f"{video_id}_metadata.json")

            # Load embedding
            embedding = np.load(embedding_path)

            # Load metadata if available
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            self.embeddings.append(embedding)
            self.metadata.append(metadata)
            self.video_ids.append(video_id)

        # Convert to numpy array
        self.embeddings = np.array(self.embeddings).astype('float32')

        print(f"Loaded {len(self.embeddings)} embeddings with shape {self.embeddings.shape}")

        if len(self.embeddings) > 0:
            self._write_table()

        return len(self.embeddings)

    def _write_table(self):
        """Write the embeddings and metadata to a LanceDB table, replacing any earlier copy."""
        dimension = self.embeddings.shape[1]
        schema = pa.schema([
            pa.field("video_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
            # Metadata varies by source (video or image), so it is stored as a JSON string
            pa.field("metadata", pa.string()),
        ])
        data = pa.table({
            "video_id": self.video_ids,
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(self.embeddings.ravel(), pa.float32()), dimension),
            "metadata": [json.dumps(m) for m in self.metadata],
        }, schema=schema)

        self.db = lancedb.connect(self.db_uri)
        self.table = self.db.create_table(TABLE_NAME, data=data, schema=schema, mode="overwrite")
        print(f"Wrote {self.table.count_rows()} rows to LanceDB table '{TABLE_NAME}' at {self.db_uri}")

    def build_index(self, index_type="Flat"):
        """Build a LanceDB vector index for fast similarity search."""
        if len(self.embeddings) == 0:
            print("No embeddings loaded. Call load_embeddings() first.")
            return False

        if index_type not in INDEX_TYPES:
            raise ValueError(f"Unsupported index type: {index_type}")

        start_time = time.time()

        lance_index = INDEX_TYPES[index_type]
        if lance_index is not None:
            # About sqrt(n) partitions, and never more partitions than vectors
            num_partitions = max(1, min(len(self.embeddings), int(np.sqrt(len(self.embeddings)))))
            self.table.create_index(metric="l2", index_type=lance_index,
                                    num_partitions=num_partitions, replace=True)

        self.index_type = index_type
        self.index = self.table

        build_time = time.time() - start_time

        print(f"Built {index_type} index with {len(self.embeddings)} vectors in {build_time:.3f} seconds")

        return True

    def find_similar(self, query_vector, k=5):
        """Find k most similar videos to the query vector."""
        if self.index is None:
            print("Index not built. Call build_index() first.")
            return []

        # Ensure query is a float32 vector
        query = np.asarray(query_vector, dtype='float32').ravel()

        start_time = time.time()

        # Search the table. Squared L2 distance, the same measure as FAISS IndexFlatL2
        search = self.table.search(query).distance_type("l2").limit(k)
        if INDEX_TYPES[self.index_type] is None:
            search = search.bypass_vector_index()
        rows = search.select(["video_id", "metadata", "_distance"]).to_list()

        search_time = time.time() - start_time

        # Format results
        results = []
        for i, row in enumerate(rows):
            distance = float(row["_distance"])
            result = {
                "rank": i + 1,
                "video_id": row["video_id"],
                "distance": distance,
                "similarity_score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity
            }

            # Add metadata if available
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            if metadata:
                result.update({
                    "metadata": metadata
                })

            results.append(result)

        # Add search stats
        search_stats = {
            "search_time_ms": round(search_time * 1000, 3),
            "num_results": len(results),
            "timestamp": datetime.now().isoformat(),
            "index_size": len(self.embeddings)
        }

        return results, search_stats

    def find_similar_by_id(self, video_id, k=5):
        """Find k most similar videos to the specified video ID."""
        if video_id not in self.video_ids:
            print(f"Video ID {video_id} not found in the index.")
            return [], {}
        
        # Get the embedding for the specified video ID
        idx = self.video_ids.index(video_id)
        query_vector = self.embeddings[idx]
        
        return self.find_similar(query_vector, k)
    
    def batch_search(self, query_file, output_file, k=5):
        """Perform batch similarity search for multiple queries."""
        if not os.path.exists(query_file):
            print(f"Query file {query_file} not found.")
            return False
        
        with open(query_file, 'r') as f:
            queries = json.load(f)
        
        print(f"Processing {len(queries)} search queries...")
        
        results = []
        batch_stats = {
            "total_queries": len(queries),
            "start_time": datetime.now().isoformat(),
            "total_search_time_ms": 0,
            "index_size": len(self.embeddings)
        }
        
        for query in queries:
            query_id = query.get("query_id", "unknown")
            query_type = query.get("query_type", "vector")
            
            if query_type == "vector" and "vector" in query:
                query_vector = np.array(query["vector"]).astype('float32')
                similar_results, stats = self.find_similar(query_vector, k)
            elif query_type == "video_id" and "video_id" in query:
                similar_results, stats = self.find_similar_by_id(query["video_id"], k)
            else:
                print(f"Invalid query format for query_id {query_id}")
                continue
            
            batch_stats["total_search_time_ms"] += stats["search_time_ms"]
            
            results.append({
                "query_id": query_id,
                "query_type": query_type,
                "search_time_ms": stats["search_time_ms"],
                "results": similar_results
            })
        
        batch_stats["end_time"] = datetime.now().isoformat()
        batch_stats["average_search_time_ms"] = batch_stats["total_search_time_ms"] / len(queries) if queries else 0
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump({
                "batch_stats": batch_stats,
                "results": results
            }, f, indent=4)
        
        print(f"Batch search completed. Results saved to {output_file}")
        print(f"Average search time: {batch_stats['average_search_time_ms']:.3f} ms per query")
        
        return True
    
    def analyze_embeddings(self):
        """Analyze the embeddings to find clusters and outliers."""
        if len(self.embeddings) == 0:
            print("No embeddings loaded. Call load_embeddings() first.")
            return {}
        
        # Compute pairwise distances for all embeddings
        n = len(self.embeddings)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Compute Euclidean distance
                dist = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Find potential clusters
        # Simple approach: consider videos with distance < threshold as part of same cluster
        threshold = np.percentile(distances, 25)  # Use 25th percentile as threshold
        
        clusters = defaultdict(list)
        cluster_id = 0
        
        for i in range(n):
            # Find all videos close to this one
            similar_indices = [j for j in range(n) if distances[i, j] < threshold and i != j]
            
            if similar_indices:
                # Check if this video is already in a cluster
                found_cluster = False
                for cid, members in clusters.items():
                    if i in members:
                        # Add the similar videos to this cluster
                        for j in similar_indices:
                            if j not in members:
                                clusters[cid].append(j)
                        found_cluster = True
                        break
                
                if not found_cluster:
                    # Create a new cluster
                    clusters[cluster_id] = [i] + similar_indices
                    cluster_id += 1
        
        # Format cluster results
        cluster_results = []
        for cid, members in clusters.items():
            cluster_videos = [self.video_ids[i] for i in members]
            avg_distance = np.mean([distances[i, j] for i in members for j in members if i != j])
            
            cluster_results.append({
                "cluster_id": cid,
                "size": len(members),
                "videos": cluster_videos,
                "average_distance": float(avg_distance),
                "cohesion": float(1.0 / (1.0 + avg_distance))  # Higher cohesion means more similar
            })
        
        # Find outliers (videos far from all others)
        avg_distances = np.mean(distances, axis=1)
        outlier_threshold = np.percentile(avg_distances, 75)  # Use 75th percentile as threshold
        
        outliers = []
        for i, avg_dist in enumerate(avg_distances):
            if avg_dist > outlier_threshold:
                outliers.append({
                    "video_id": self.video_ids[i],
                    "average_distance": float(avg_dist),
                    "isolation_score": float(avg_dist / np.mean(avg_distances))  # How much more isolated than average
                })
        
        # Sort outliers by isolation score (most isolated first)
        outliers.sort(key=lambda x: x["isolation_score"], reverse=True)
        
        analysis = {
            "num_embeddings": n,
            "num_clusters": len(clusters),
            "average_distance": float(np.mean(distances)),
            "clusters": cluster_results,
            "outliers": outliers,
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Similarity search for video embeddings")
    parser.add_argument("--embeddings", required=True, help="Directory containing embedding files")
    parser.add_argument("--action", choices=["search", "batch", "analyze"], default="search",
                        help="Action to perform")
    parser.add_argument("--query", help="Query video ID for search action")
    parser.add_argument("--query-file", help="JSON file with queries for batch search")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--index", choices=list(INDEX_TYPES), default="Flat",
                        help="Index type to use")
    
    args = parser.parse_args()
    
    search = SimilaritySearch(args.embeddings)
    search.load_embeddings()
    search.build_index(index_type=args.index)
    
    if args.action == "search" and args.query:
        results, stats = search.find_similar_by_id(args.query, k=args.k)
        
        print(f"Found {len(results)} similar videos to {args.query} in {stats['search_time_ms']} ms")
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result['video_id']} - Similarity: {result['similarity_score']:.3f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "query": args.query,
                    "results": results,
                    "stats": stats
                }, f, indent=4)
    
    elif args.action == "batch" and args.query_file and args.output:
        search.batch_search(args.query_file, args.output, k=args.k)
    
    elif args.action == "analyze":
        analysis = search.analyze_embeddings()
        
        print(f"Found {len(analysis['clusters'])} clusters and {len(analysis['outliers'])} outliers")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=4)
        else:
            print("Clusters:")
            for cluster in analysis['clusters']:
                print(f"  Cluster {cluster['cluster_id']}: {len(cluster['videos'])} videos, "
                      f"cohesion: {cluster['cohesion']:.3f}")
            
            print("\nTop 5 Outliers:")
            for outlier in analysis['outliers'][:5]:
                print(f"  {outlier['video_id']} - Isolation score: {outlier['isolation_score']:.3f}")
    
    else:
        print("Invalid action or missing required arguments.")
        parser.print_help()

if __name__ == "__main__":
    main() 