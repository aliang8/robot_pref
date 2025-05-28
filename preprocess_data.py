"""Preprocess trajectory data for preference learning."""

import itertools
import os
import pickle
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import utils.dtw as dtw
from models.image_embedder import ImageEmbedder
from utils.data import load_tensordict, segment_episodes_dynamic
from utils.seed import set_seed


def compute_dtw_distance_matrix(segments: List[Dict], use_relative_eef: bool, dtw_type: str = "dtw") -> np.ndarray:
    """Compute DTW distance matrix between segments.

    Args:
        segments: List of segments to compute distances between
        use_relative_eef: Whether to use relative EEF positions
        dtw_type: Type of DTW to compute ("dtw" or "sdtw")

    Returns:
        distance_matrix: Matrix of DTW distances between segments
    """
    n_segments = len(segments)
    print(f"Computing {dtw_type.upper()} distance matrix for {n_segments} segments")

    distance_matrix = np.zeros((n_segments, n_segments))
    min_dist = float("inf")
    max_dist = float("-inf")
    sum_dist = 0
    count = 0
    non_finite_count = 0

    total_comparisons = n_segments * (n_segments - 1) // 2
    with tqdm(total=total_comparisons, desc=f"Computing {dtw_type.upper()} distances") as pbar:
        for i in range(n_segments):
            for j in range(i + 1, n_segments):
                # EE positions
                query = segments[i]["obs"].numpy()[:, :3]
                reference = segments[j]["obs"].numpy()[:, :3]

                # Relative
                if use_relative_eef:
                    query = query[1:] - query[:-1]
                    reference = reference[1:] - reference[:-1]

                if dtw_type == "sdtw":
                    cost, _ = dtw.get_single_match_subsequence(query, reference)
                else:  # dtw_type == "dtw"
                    cost, _ = dtw.get_single_match(query, reference)

                distance_matrix[i, j] = cost
                distance_matrix[j, i] = cost

                # Update statistics
                if np.isfinite(cost):
                    min_dist = min(min_dist, cost)
                    max_dist = max(max_dist, cost)
                    sum_dist += cost
                    count += 1
                else:
                    non_finite_count += 1

                pbar.update(1)

    if count > 0:
        avg_dist = sum_dist / count
        print(
            f"{dtw_type.upper()} distance statistics - Min: {min_dist:.2f}, Max: {max_dist:.2f}, Avg: {avg_dist:.2f}"
        )
    if non_finite_count > 0:
        print(
            f"WARNING: {non_finite_count} {dtw_type.upper()} distances used fallback due to non-finite values"
        )
    return distance_matrix


def compute_image_embeddings(
    images: torch.Tensor,
    embedder: ImageEmbedder,
    batch_size: int,
    device: str
) -> torch.Tensor:
    """Compute image embeddings in batches.

    Args:
        images: Image tensor of shape [N, H, W, C] or [N, C, H, W]
        embedder: Image embedding model
        batch_size: Batch size for processing
        device: Device to use for computation

    Returns:
        embeddings: Tensor of image embeddings
    """
    embedder.eval()
    
    # Check and convert image format
    if images.shape[-1] == 3:  # If last dimension is 3, it's in HWC format
        print("Converting images from HWC to CHW format")
        images = images.permute(0, 3, 1, 2)
    
    # Normalize images to [0, 1] if needed
    if images.max() > 1.0:
        print("Normalizing images from [0, 255] to [0, 1] range")
        images = images.float() / 255.0
    
    print(f"Image shape after preprocessing: {images.shape}, Range: [{images.min():.3f}, {images.max():.3f}]")
    
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Computing image embeddings"):
            batch = images[i:i + batch_size].to(device)
            embedding = embedder(batch)
            if isinstance(embedding, dict):
                embedding = embedding["feature_map"]
            if embedding.ndim > 2:  # If features are spatial, take mean
                embedding = embedding.mean(dim=[-2, -1]) if embedding.ndim == 4 else embedding.mean(dim=1)
            embeddings.append(embedding.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    print(f"Final embedding shape: {embeddings.shape}")
    return embeddings


@hydra.main(config_path="config", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    print("\n" + "=" * 50)
    print("Preprocessing trajectory data")
    print("=" * 50)

    # Print config
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    set_seed(cfg.seed)
    
    # Set up input and output paths
    data_path = Path(cfg.data.data_path)
    output_dir = data_path.parent
    print(f"\nOutput directory set to: {output_dir}")

    # Load data
    print(f"\nLoading data from {data_path}")
    data = load_tensordict(data_path)
    data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    # Segment episodes
    print(f"\nSegmenting episodes with length {cfg.data.segment_length}")
    segments, segment_indices = segment_episodes_dynamic(data, cfg.data.segment_length)
    
    # Create segment pairs
    segment_indices_array = np.array(segment_indices)
    segment_pairs = list(itertools.combinations(range(len(segments)), 2))
    segment_pairs = np.array(segment_pairs)

    # Save segment indices and pairs
    np.save(output_dir / 'segment_start_end_indices.npy', segment_indices_array)
    np.save(output_dir / 'segment_pairs.npy', segment_pairs)
    print(f"Saved segment indices and pairs to {output_dir}")

    # Compute DTW matrices if enabled
    if cfg.dtw.enabled:
        if cfg.dtw.use_subsequence:
            dtw_matrix_file = output_dir / f"sdtw_matrix_{cfg.data.segment_length}.pkl"
        else:
            dtw_matrix_file = output_dir / f"dtw_matrix_{cfg.data.segment_length}.pkl"

        
        if os.path.exists(dtw_matrix_file) and not cfg.data.overwrite:
            print(f"DTW matrix file already exists: {dtw_matrix_file}")
        else:
            print("\nComputing DTW distance matrix...")
            dtw_matrix = compute_dtw_distance_matrix(segments, cfg.dtw.use_relative_eef, "dtw")
            
            print(f"Saving DTW matrix to: {dtw_matrix_file}")
            with open(dtw_matrix_file, "wb") as f:
                pickle.dump((dtw_matrix, segment_indices), f)

        # Compute S-DTW matrix
        sdtw_matrix_file = output_dir / f"sdtw_matrix_{cfg.data.segment_length}.pkl"
        
        if os.path.exists(sdtw_matrix_file) and not cfg.data.overwrite:
            print(f"S-DTW matrix file already exists: {sdtw_matrix_file}")
        else:
            print("\nComputing S-DTW distance matrix...")
            sdtw_matrix = compute_dtw_distance_matrix(segments, cfg.dtw.use_relative_eef, "sdtw")
            
            print(f"Saving S-DTW matrix to: {sdtw_matrix_file}")
            with open(sdtw_matrix_file, "wb") as f:
                pickle.dump((sdtw_matrix, segment_indices), f)

    # Compute image embeddings if enabled
    if cfg.image_embedding.enabled and "image" in data:
        print("\nComputing image embeddings...")
        print(f"Original image shape: {data['image'].shape}")
        
        # Initialize image embedder
        embedder = ImageEmbedder(
            model_name=cfg.image_embedding.model_name,
            device=cfg.image_embedding.device,
            feature_fmt=cfg.image_embedding.feature_fmt,
            use_spatial_features=cfg.image_embedding.use_spatial_features
        )
        embedder = embedder.to(cfg.image_embedding.device)

        # Compute embeddings
        images = data["image"]
        image_embeddings = compute_image_embeddings(
            images,
            embedder,
            cfg.image_embedding.batch_size,
            cfg.image_embedding.device
        )
        
        # Create new dataset with embeddings
        embedded_data = {k: v for k, v in data.items()}
        embedded_data["image_embedding"] = image_embeddings
        del embedded_data["image"]  # Remove original images to save space
        
        # Save embedded dataset
        dataset_name = data_path.stem
        output_path = output_dir / f"{dataset_name}_embedded.pt"
        torch.save(embedded_data, output_path)
        print(f"Saved embedded dataset to: {output_path}")

    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main() 