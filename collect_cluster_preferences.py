import torch
import numpy as np
import os
import random
import pickle
import argparse
import itertools
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
from IPython.display import HTML, display

# For running in interactive environments
is_notebook = 'ipykernel' in sys.modules
if is_notebook:
    from IPython.display import display, HTML
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Import utility functions
from trajectory_utils import (
    DEFAULT_DATA_PATHS,
    RANDOM_SEED,
    load_tensordict,
    load_preprocessed_segments
)

# Import clustering functions
from eef_clustering import extract_eef_trajectories

# Import visualization functions from eef_segment_matching instead
from eef_segment_matching import (
    create_eef_trajectory_animation,
    create_comparison_video
)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_clustering_results(results_file):
    """Load clustering results from a file."""
    print(f"Loading clustering results from {results_file}")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results

def get_representative_segments(data, clusters, segment_indices, n_per_cluster=1):
    """Extract representative segments for each cluster.
    
    Args:
        data: TensorDict with observations
        clusters: Cluster assignments
        segment_indices: List of (start_idx, end_idx) for each segment
        n_per_cluster: Number of representative segments per cluster
        
    Returns:
        dict: Mapping from cluster_id to list of representative segments
              Each segment is (segment_idx, start_idx, end_idx)
    """
    unique_clusters = np.unique(clusters)
    print(f"Found {len(unique_clusters)} unique clusters")
    
    # Get cluster centroids by finding segment closest to mean
    cluster_representatives = {}
    
    # For each cluster, find representative segments
    for cluster_id in unique_clusters:
        # Get indices of segments in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        
        # Skip if cluster is empty
        if len(cluster_indices) == 0:
            print(f"Cluster {cluster_id} is empty")
            continue
            
        print(f"Cluster {cluster_id}: {len(cluster_indices)} segments")
        
        # Choose representative segments
        # Strategy 1: Random sampling
        n_samples = min(n_per_cluster, len(cluster_indices))
        representative_indices = random.sample(list(cluster_indices), n_samples)
        
        # Save representative segments with metadata
        representatives = []
        for idx in representative_indices:
            start_idx, end_idx = segment_indices[idx]
            representatives.append((idx, start_idx, end_idx))
            
        cluster_representatives[cluster_id] = representatives
    
    return cluster_representatives

def display_segment(data, start_idx, end_idx, title=None, cluster_id=None):
    """Display a single segment."""
    # Create animation for EEF trajectory
    eef_positions = data["obs"][:, :3]
    
    if title is None:
        title = f"Segment from Cluster {cluster_id}"
        
    anim = create_eef_trajectory_animation(eef_positions, start_idx, end_idx, title=title)
    
    # Display animation
    if is_notebook:
        display(HTML(anim.to_jshtml()))
    
    return anim

def present_preference_query(data, segment1, segment2, query_id=None, skip_videos=False):
    """Present a preference query to the user.
    
    Args:
        data: TensorDict with observations
        segment1: (idx1, start_idx1, end_idx1) for first segment
        segment2: (idx2, start_idx2, end_idx2) for second segment
        query_id: Optional ID for the query
        skip_videos: If True, skip generating videos to save time
        
    Returns:
        int: 1 if segment1 is preferred, 2 if segment2 is preferred, 0 if equal
    """
    idx1, start_idx1, end_idx1 = segment1
    idx2, start_idx2, end_idx2 = segment2
    
    # Clear output in notebook environments
    if is_notebook:
        from IPython.display import clear_output
        clear_output(wait=True)
    
    query_title = f"Query {query_id}" if query_id is not None else "Preference Query"
    print(f"\n{query_title}")
    print("=" * 40)
    
    # Get end-effector trajectories
    eef_positions = data["obs"][:, :3]
    
    # Display the two segments textually
    print(f"Segment 1: {start_idx1}-{end_idx1} (Length: {end_idx1-start_idx1+1})")
    print(f"Segment 2: {start_idx2}-{end_idx2} (Length: {end_idx2-start_idx2+1})")
    
    # Display animations and video only if not skipping
    if not skip_videos:
        print("\nCreating trajectory animations...")
        
        print("Segment 1:")
        anim1 = create_eef_trajectory_animation(eef_positions, start_idx1, end_idx1, title="Segment 1")
        
        print("Segment 2:")
        anim2 = create_eef_trajectory_animation(eef_positions, start_idx2, end_idx2, title="Segment 2")
        
        if is_notebook:
            display(HTML(anim1.to_jshtml()))
            display(HTML(anim2.to_jshtml()))
        
        # If running in interactive mode, create side-by-side video
        temp_video = "temp_comparison.mp4"
        create_comparison_video(
            eef_positions,
            (start_idx1, end_idx1),
            [(start_idx2, end_idx2)], 
            [0],  # Placeholder for distance
            dataset_indicators=None,
            output_file=temp_video,
            data=data if "image" in data else None
        )
        
        print("\nCreated comparison video:", temp_video)
    else:
        print("\n[Videos skipped to save time]")
    
    # Get user preference
    while True:
        print("\nWhich segment do you prefer?")
        print("1: Segment 1")
        print("2: Segment 2")
        print("0: Equal/Cannot decide")
        
        try:
            preference = input("Enter preference (0/1/2): ").strip()
            preference = int(preference)
            if preference in [0, 1, 2]:
                break
            else:
                print("Invalid input. Please enter 0, 1, or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    return preference

def collect_cluster_preferences(data, cluster_representatives, num_comparisons=None, skip_videos=False):
    """Collect user preferences between cluster representatives.
    
    Args:
        data: TensorDict with observations
        cluster_representatives: Dict mapping cluster_id to representative segments
        num_comparisons: Number of comparisons to conduct (default: all pairs)
        skip_videos: Skip generating videos for preference collection
        
    Returns:
        list: User preferences in format [(cluster_id1, cluster_id2, preference), ...]
              where preference is 1 if cluster_id1 is preferred, 2 if cluster_id2 is preferred
    """
    # Get all cluster IDs
    cluster_ids = sorted(cluster_representatives.keys())
    print(f"Collecting preferences for {len(cluster_ids)} clusters")
    
    # Generate all possible pairs of cluster IDs
    cluster_pairs = list(itertools.combinations(cluster_ids, 2))
    
    # Shuffle pairs for randomness
    random.shuffle(cluster_pairs)
    
    # If num_comparisons is specified, limit the number of comparisons
    if num_comparisons and num_comparisons < len(cluster_pairs):
        print(f"Limiting to {num_comparisons} comparisons out of {len(cluster_pairs)} possible pairs")
        cluster_pairs = cluster_pairs[:num_comparisons]
    
    # Collect preferences
    preferences = []
    
    for i, (cluster_id1, cluster_id2) in enumerate(cluster_pairs):
        print(f"\nComparison {i+1}/{len(cluster_pairs)}: Cluster {cluster_id1} vs Cluster {cluster_id2}")
        
        # Select a random representative from each cluster
        segment1 = random.choice(cluster_representatives[cluster_id1])
        segment2 = random.choice(cluster_representatives[cluster_id2])
        
        # Present preference query
        preference = present_preference_query(data, segment1, segment2, query_id=i+1, skip_videos=skip_videos)
        
        # Record preference
        preferences.append((cluster_id1, cluster_id2, preference))
        print(f"Recorded preference: {preference}")
    
    return preferences

def derive_cluster_ranking(preferences):
    """Derive an approximate ranking of clusters from pairwise preferences.
    
    Args:
        preferences: List of (cluster_id1, cluster_id2, preference) tuples
        
    Returns:
        list: Ordered list of cluster IDs from most to least preferred
    """
    print("Deriving cluster ranking from preferences...")
    
    # Extract all unique cluster IDs
    cluster_ids = set()
    for c1, c2, _ in preferences:
        cluster_ids.add(c1)
        cluster_ids.add(c2)
    
    cluster_ids = sorted(list(cluster_ids))
    
    # Initialize win counts for each cluster
    win_counts = {cluster_id: 0 for cluster_id in cluster_ids}
    
    # Count wins for each cluster
    for c1, c2, pref in preferences:
        if pref == 1:
            win_counts[c1] += 1
        elif pref == 2:
            win_counts[c2] += 1
    
    # Sort clusters by win count (descending)
    ranked_clusters = sorted(win_counts.keys(), key=lambda c: win_counts[c], reverse=True)
    
    print("Cluster ranking (most to least preferred):")
    for i, cluster_id in enumerate(ranked_clusters):
        print(f"  {i+1}. Cluster {cluster_id} (wins: {win_counts[cluster_id]})")
    
    return ranked_clusters

def generate_augmented_preferences(data, clusters, segment_indices, cluster_ranking):
    """Generate augmented preferences based on cluster ranking.
    
    Args:
        data: TensorDict with observations
        clusters: Cluster assignments
        segment_indices: List of (start_idx, end_idx) for each segment
        cluster_ranking: Ordered list of cluster IDs from most to least preferred
        
    Returns:
        list: Augmented preferences [(i, j, pref), ...]
              where i, j are segment indices and pref is 1 if i preferred, 2 if j preferred
    """
    print("Generating augmented preferences...")
    
    # Convert cluster ranking to a mapping of cluster to rank (0 is best)
    cluster_ranks = {cluster_id: rank for rank, cluster_id in enumerate(cluster_ranking)}
    
    # Generate augmented preferences
    augmented_preferences = []
    
    # Iterate through all possible segment pairs
    n_segments = len(segment_indices)
    segments_by_cluster = {c: [] for c in cluster_ranking}
    
    # Group segments by cluster
    for i, (seg_idx, cluster) in enumerate(zip(range(n_segments), clusters)):
        segments_by_cluster[cluster].append(seg_idx)
        
    # Print distribution
    print("Segments per cluster:")
    for cluster_id, seg_indices in segments_by_cluster.items():
        print(f"  Cluster {cluster_id}: {len(seg_indices)} segments")
    
    # Generate preferences from cluster ranking
    # For each pair of clusters where one has higher rank than the other
    augmented_count = 0
    
    # Create a progress bar for the outer loop
    for i, cluster1 in enumerate(tqdm(cluster_ranking, desc="Processing clusters")):
        for cluster2 in cluster_ranking[i+1:]:  # Lower ranks (less preferred)
            # Get segments from both clusters
            segments1 = segments_by_cluster[cluster1]
            segments2 = segments_by_cluster[cluster2]
            
            # If both clusters have segments
            if segments1 and segments2:
                # Generate a subset of all possible pairs to avoid excessive augmentation
                # Determine sampling ratio based on dataset size
                total_pairs = len(segments1) * len(segments2)
                
                # Adjust sampling based on total pairs
                if total_pairs > 1000:
                    # Sample a fixed number or percentage
                    num_samples = min(1000, int(total_pairs * 0.1))
                else:
                    num_samples = total_pairs
                
                # Sample segment pairs
                pairs = []
                for _ in range(num_samples):
                    idx1 = random.choice(segments1)
                    idx2 = random.choice(segments2)
                    pairs.append((idx1, idx2))
                
                # Generate preferences
                for idx1, idx2 in pairs:
                    # Cluster1 has higher rank (lower rank number) than Cluster2
                    augmented_preferences.append((idx1, idx2, 1))  # Prefer segment from cluster1
                    augmented_count += 1
    
    print(f"Generated {augmented_count} augmented preferences")
    
    return augmented_preferences

def create_preference_dataset(data, segment_indices, augmented_preferences, output_file):
    """Create a dataset for training a reward model from augmented preferences.
    
    Args:
        data: TensorDict with observations
        segment_indices: List of (start_idx, end_idx) for each segment
        augmented_preferences: List of (i, j, pref) preference tuples
        output_file: Path to save the dataset
        
    Returns:
        dict: Preference dataset with segments and labels
    """
    print(f"Creating preference dataset at {output_file}")
    
    # Extract segments from data
    all_segments = []
    for start_idx, end_idx in segment_indices:
        segment_obs = data["obs"][start_idx:end_idx+1]
        all_segments.append(segment_obs)
    
    # Create preference pairs
    segment_pairs = []
    preference_labels = []
    
    for i, j, pref in augmented_preferences:
        segment_pairs.append((i, j))
        preference_labels.append(pref)
    
    # Create dataset
    preference_dataset = {
        'segments': all_segments,
        'segment_indices': segment_indices,
        'segment_pairs': segment_pairs,
        'preference_labels': preference_labels,
        'metadata': {
            'n_segments': len(all_segments),
            'n_preferences': len(preference_labels),
            'data_path': data.get('_source_path', 'unknown')
        }
    }
    
    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(preference_dataset, f)
    
    print(f"Saved preference dataset with {len(preference_labels)} preferences")
    
    return preference_dataset

def main():
    parser = argparse.ArgumentParser(description="Collect and augment preferences based on cluster representatives")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATHS[0],
                        help="Path to the PT file containing trajectory data")
    parser.add_argument("--preprocessed_data", type=str, default=None,
                        help="Path to preprocessed data file (used instead of data_path if provided)")
    parser.add_argument("--clustering_results", type=str, required=True,
                        help="Path to the pickle file with clustering results")
    parser.add_argument("--output_dir", type=str, default="preference_data",
                        help="Directory to save preference data and dataset")
    parser.add_argument("--n_representatives", type=int, default=3,
                        help="Number of representative segments per cluster")
    parser.add_argument("--max_comparisons", type=int, default=None,
                        help="Maximum number of pairwise comparisons to perform")
    parser.add_argument("--skip_videos", action="store_true",
                        help="Skip generating videos for preference collection to speed things up")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    if args.preprocessed_data:
        print(f"Loading preprocessed data from {args.preprocessed_data}")
        preproc_data = load_preprocessed_segments(args.preprocessed_data)
        
        # Create data structure expected by preference collection functions
        data = {}
        
        # Check which fields are available in preprocessed data
        essential_fields = ['obs', 'state', 'image', 'episode', 'reward']
        missing_fields = []
        
        for field in essential_fields:
            if field in preproc_data:
                data[field] = preproc_data[field]
            else:
                missing_fields.append(field)
        
        # Handle missing fields
        if 'obs' not in data and 'state' not in data:
            print(f"WARNING: Preprocessed data does not contain observation data (obs or state).")
            print(f"Loading original data from {args.data_path} for observations.")
            orig_data = load_tensordict(args.data_path)
            
            # Copy missing essential fields
            for field in missing_fields:
                if field in orig_data:
                    data[field] = orig_data[field]
                    print(f"Loaded field '{field}' from original data")
        
        # Add source path for reference
        data['_source_path'] = args.preprocessed_data
    else:
        print(f"Loading data from {args.data_path}")
        data = load_tensordict(args.data_path)
        data['_source_path'] = args.data_path
    
    # Load clustering results
    clustering_results = load_clustering_results(args.clustering_results)
    clusters = clustering_results['clusters']
    segment_indices = clustering_results['segment_indices']
    
    print(f"Loaded clustering results with {len(clusters)} clusters")
    
    # Get representative segments for each cluster
    cluster_representatives = get_representative_segments(
        data, clusters, segment_indices, n_per_cluster=args.n_representatives
    )
    
    # Collect user preferences between clusters
    user_preferences = collect_cluster_preferences(
        data, cluster_representatives, 
        num_comparisons=args.max_comparisons,
        skip_videos=args.skip_videos
    )
    
    # Save raw preferences
    preferences_file = os.path.join(args.output_dir, "raw_preferences.pkl")
    with open(preferences_file, 'wb') as f:
        pickle.dump({
            'user_preferences': user_preferences,
            'cluster_representatives': cluster_representatives
        }, f)
    
    print(f"Saved raw preferences to {preferences_file}")
    
    # Derive cluster ranking
    cluster_ranking = derive_cluster_ranking(user_preferences)
    
    # Generate augmented preferences
    augmented_preferences = generate_augmented_preferences(
        data, clusters, segment_indices, cluster_ranking
    )
    
    # Create preference dataset
    dataset_file = os.path.join(args.output_dir, "preference_dataset.pkl")
    preference_dataset = create_preference_dataset(
        data, segment_indices, augmented_preferences, dataset_file
    )
    
    print("Process complete!")
    print(f"Collected {len(user_preferences)} direct user preferences")
    print(f"Generated {len(augmented_preferences)} augmented preferences")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 