#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb


def fetch_wandb_runs(project="robot_pref", entity="clvr", filters=None, max_runs=None):
    """Fetch runs from wandb.
    
    Args:
        project: Name of the wandb project
        entity: Name of the wandb entity
        filters: Dictionary of filters to apply to the runs
        max_runs: Maximum number of runs to fetch
        
    Returns:
        List of run data dictionaries
    """
    print(f"Fetching runs from wandb project: {entity}/{project}")
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Store filter info for manual filtering later
    user_filter = filters.get("user") if filters else None
    if user_filter:
        print(f"Will filter for user: {user_filter}")
    
    # Fetch all runs
    print(f"Fetching all runs from {entity}/{project}...")
    runs = api.runs(f"{entity}/{project}")
    
    print(f"Found {len(runs)} total runs")
    
    # Filter runs manually
    filtered_runs = []
    for run in runs:
        include_run = True
        
        # Apply user filter if specified
        if user_filter:
            # Try to get user info
            if hasattr(run, "user") and run.user.username != user_filter:
                include_run = False
        
        if include_run:
            filtered_runs.append(run)
    
    print(f"After filtering: {len(filtered_runs)} runs")
    
    # Use filtered runs
    runs = filtered_runs
    
    # Limit number of runs if specified
    if max_runs is not None and len(runs) > max_runs:
        runs = runs[:max_runs]
        print(f"Limited to {len(runs)} runs")
    
    # Extract run data
    run_data = []
    for run in runs:
        # Basic run information
        run_dict = {
            "id": run.id,
            "name": run.name,
            "tags": run.tags,
            "state": run.state,
            "created_at": run.created_at,
            "config": run.config,
            "summary": run.summary._json_dict,
            "url": run.url
        }
        
        # Add user information if available
        if hasattr(run, "user"):
            run_dict["user"] = run.user.username

        # Get history data - focusing only on eval/success_rate
        # TODO: fix
        history = run.history(keys=["eval/success_rate", "eval/epoch"])
        if history.empty:
            history = run.history(keys=["eval/success_rate", "epoch"])
        
        if not history.empty and "eval/success_rate" in history.columns:
            # Get success rates, filtering out NaN values
            success_rates = history["eval/success_rate"].dropna().tolist()
            epochs = history["epoch"].dropna().tolist() if "epoch" in history.columns else list(range(len(success_rates)))
            
            run_dict["history"] = {
                "eval_success_rates": success_rates,
                "epochs": epochs
            }
            
            # Add run to our data if it has success rates
            if success_rates:
                print(f"Run {run.name}: Found {len(success_rates)} success rate values")
                run_data.append(run_dict)
            else:
                print(f"Run {run.name}: No success rate values found")
        else:
            print(f"Run {run.name}: No eval/success_rate in history")
    
    print(f"Extracted data from {len(run_data)} runs with success rate metrics")
    return run_data

def create_run_dataframe(run_data):
    """Convert run data to a pandas DataFrame with relevant metrics.
    
    Args:
        run_data: List of run data dictionaries
        
    Returns:
        Pandas DataFrame with run information
    """
    rows = []
    
    for run in run_data:
        # Extract basic run info
        row = {
            "run_id": run["id"],
            "run_name": run["name"],
            "state": run["state"],
            "created_at": run["created_at"],
            "url": run["url"],
            "user": run.get("user", "unknown")
        }
        
        # Extract config information
        if "config" in run:
            config = run["config"]
            
            # Algorithm and dataset
            row["algorithm"] = config.get("algorithm", "unknown")
            if "data" in config and "data_path" in config["data"]:
                path_obj = Path(config["data"]["data_path"])
                row["dataset"] = f"{path_obj.parent.name}_{path_obj.stem}"
            else:
                row["dataset"] = "unknown"
                
            # Reward model information
            if "data" in config and "reward_model_path" in config["data"]:
                row["reward_model"] = config["data"].get("reward_model_path", "none")
                row["use_ground_truth"] = config["data"].get("use_ground_truth", False)
                row["scale_rewards"] = config["data"].get("scale_rewards", False)
            
            # Training parameters
            if "training" in config:
                row["n_epochs"] = config["training"].get("n_epochs", 0)
                
            # Model parameters
            if "model" in config:
                row["actor_lr"] = config["model"].get("actor_learning_rate", 0)
                row["critic_lr"] = config["model"].get("critic_learning_rate", 0)
                row["expectile"] = config["model"].get("expectile", 0)
                row["weight_temp"] = config["model"].get("weight_temp", 0)
            
            # Random seed
            row["random_seed"] = config.get("random_seed", 42)
        
        # Add history data for success rates
        if "history" in run and run["history"]["eval_success_rates"]:
            success_rates = run["history"]["eval_success_rates"]
            
            # Get the latest success rate
            row["latest_eval_success_rate"] = success_rates[-1]
            
            # Get the maximum success rate
            row["max_eval_success_rate"] = max(success_rates)
            
            # Calculate average of top 3 success rates
            top_3_rates = sorted(success_rates, reverse=True)[:3]
            row["top3_avg_eval_success_rate"] = sum(top_3_rates) / len(top_3_rates)
            
            # Store individual top 3 values
            for i, rate in enumerate(top_3_rates[:3]):
                row[f"top_eval_success_{i+1}"] = rate
            
            # Store full history for plotting
            row["history_eval_success_rates"] = success_rates
            row["history_epochs"] = run["history"]["epochs"]
        
        rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by created_at (newest first)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at", ascending=False)
    
    return df

def save_summary_csv(df, output_path="iql_runs_summary.csv"):
    """Save a summary CSV file with run information.
    
    Args:
        df: DataFrame with run information
        output_path: Path to save the CSV file
    """
    # Select columns to include in the summary
    columns = [
        "run_name", "user", "dataset", "algorithm", "reward_model", "use_ground_truth", "scale_rewards",
        "n_epochs", "actor_lr", "critic_lr", "expectile", "weight_temp", "random_seed",
        "top3_avg_eval_success_rate", "top_eval_success_1", "top_eval_success_2", "top_eval_success_3",
        "latest_eval_success_rate", "max_eval_success_rate", "created_at", "url"
    ]
    
    # Filter to only include columns that exist in the DataFrame
    existing_columns = [col for col in columns if col in df.columns]
    
    # Save to CSV
    df[existing_columns].to_csv(output_path, index=False)
    print(f"Saved summary CSV to {output_path}")
    
    # Save a specialized version with just top 3 average
    top3_columns = [
        "run_name", "user", "dataset", "algorithm", "expectile", "random_seed", 
        "top3_avg_eval_success_rate", "top_eval_success_1", "top_eval_success_2", "top_eval_success_3",
        "url"
    ]
    existing_top3_columns = [col for col in top3_columns if col in df.columns]
    top3_output_path = output_path.replace(".csv", "_top3.csv")
    df[existing_top3_columns].to_csv(top3_output_path, index=False)
    print(f"Saved top 3 summary CSV to {top3_output_path}")
    
    return top3_output_path

def plot_algorithm_comparisons(df, output_dir="algorithm_plots"):
    """Create plots comparing algorithm performance across datasets.
    
    Args:
        df: DataFrame with run information
        output_dir: Directory to save plots
        
    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make sure we have required columns
    if "algorithm" not in df.columns:
        print("'algorithm' column not found, attempting to derive from run_name...")
        # Try to extract algorithm from run_name
        if "run_name" in df.columns:
            # Common algorithms to look for in run names
            algorithms = ["bc", "iql", "cql", "sac", "td3", "ddpg"]
            
            def extract_algorithm(run_name):
                if not isinstance(run_name, str):
                    return "unknown"
                run_name = run_name.lower()
                for alg in algorithms:
                    if alg in run_name:
                        return alg
                return "unknown"
            
            df["algorithm"] = df["run_name"].apply(extract_algorithm)
            print(f"Extracted algorithms: {df['algorithm'].unique()}")
    
    # Filter out rows with NaN success rates
    df_filtered = df.dropna(subset=["top3_avg_eval_success_rate"])
    
    if len(df_filtered) == 0:
        print("No runs with valid success rate metrics found. Cannot generate plots.")
        return output_dir
    
    print(f"Using {len(df_filtered)} runs with valid metrics out of {len(df)} total runs")
    
    # Define metrics to plot
    metrics_to_plot = [
        {"column": "top_eval_success_1", "title": "Top 1 Checkpoint Success Rate"},
        {"column": "top_eval_success_2", "title": "Top 2 Checkpoint Success Rate"},
        {"column": "top_eval_success_3", "title": "Top 3 Checkpoint Success Rate"},
        {"column": "top3_avg_eval_success_rate", "title": "Average of Top 3 Checkpoints Success Rate"}
    ]
    
    # Get unique datasets
    datasets = df_filtered["dataset"].unique()
    
    print(f"Found {len(datasets)} datasets to plot")
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # For each metric, create plots for each dataset
    for metric in metrics_to_plot:
        metric_column = metric["column"]
        metric_title = metric["title"]
        
        # Skip if metric is not available in dataframe
        if metric_column not in df_filtered.columns:
            print(f"Metric {metric_column} not found in dataframe. Skipping.")
            continue
            
        # Group and analyze for this specific metric
        grouped = df_filtered.groupby(["dataset", "algorithm"])
        
        # Calculate statistics for the current metric
        stats = grouped[metric_column].agg([
            ("mean", np.mean),
            ("std", np.std),
            ("min", np.min),
            ("max", np.max),
            ("count", "count"),
            ("median", np.median)
        ]).reset_index()
        
        # Add 95% confidence interval
        stats["ci_95"] = 1.96 * stats["std"] / np.sqrt(stats["count"].clip(1))
        
        # Round statistics for readability
        for col in ["mean", "std", "min", "max", "median", "ci_95"]:
            stats[col] = stats[col].round(3)
        
        # Create a plot for each dataset
        for dataset in datasets:
            dataset_stats = stats[stats["dataset"] == dataset]
            
            # Skip if no data for this dataset
            if len(dataset_stats) == 0:
                continue
                
            # Sort by mean performance (descending)
            dataset_stats = dataset_stats.sort_values("mean", ascending=False)
            
            # Set up the figure
            plt.figure(figsize=(12, 6))
            
            # Create bar plot with error bars
            ax = sns.barplot(
                x="algorithm", 
                y="mean", 
                data=dataset_stats,
                palette="viridis",
                alpha=0.8
            )
            
            # Add error bars for standard deviation
            for i, row in enumerate(dataset_stats.itertuples()):
                plt.errorbar(
                    i, row.mean, 
                    yerr=row.std, 
                    fmt='none', 
                    ecolor='black', 
                    capsize=5
                )
            
            # Add data labels on top of bars
            for i, row in enumerate(dataset_stats.itertuples()):
                plt.text(
                    i, row.mean + 0.02, 
                    f"{row.mean:.3f}±{row.std:.3f}\nn={row.count}", 
                    ha='center', 
                    va='bottom',
                    fontsize=10
                )
            
            # Add a title and labels
            plt.title(f"Algorithm Performance on {dataset} Dataset\n({metric_title})")
            plt.ylabel("Success Rate")
            plt.xlabel("Algorithm")
            
            # Add gridlines for readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Adjust the y-axis to start at 0 and have some headroom
            max_value = dataset_stats["mean"].max() + dataset_stats["std"].max()
            plt.ylim(0, min(1.0, max_value * 1.2))
            
            # Save the figure
            safe_dataset = str(dataset).replace("/", "_")
            metric_short_name = metric_column.replace("top3_avg_eval_success_rate", "top3_avg").replace("top_eval_success_", "top")
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{safe_dataset}_{metric_short_name}_comparison.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved {metric_short_name} plot for {dataset} to {output_path}")
            plt.close()
    
    # Create a summary CSV with algorithm performance by dataset
    create_algorithm_summary_csv(df_filtered, output_dir)
    
    return output_dir

def create_algorithm_summary_csv(df, output_dir="algorithm_plots"):
    """Create a summary CSV with algorithm performance by dataset."""
    # Define metrics to include in summary
    metrics = [
        "top_eval_success_1",
        "top_eval_success_2", 
        "top_eval_success_3",
        "top3_avg_eval_success_rate"
    ]
    
    # Create a clean CSV with formatted values for each metric
    summary_rows = []
    
    # Group by dataset and algorithm
    grouped = df.groupby(["dataset", "algorithm"])
    
    # Get unique datasets and algorithms
    datasets = df["dataset"].unique()
    algorithms = df["algorithm"].unique()
    
    # For each dataset, create a row in the summary
    for dataset in datasets:
        row = {"Dataset": dataset}
        
        # For each algorithm, calculate statistics for each metric
        for alg in algorithms:
            group_data = df[(df["dataset"] == dataset) & (df["algorithm"] == alg)]
            
            if len(group_data) == 0:
                # No data for this combination
                for metric in metrics:
                    if metric in df.columns:
                        row[f"{alg}_{metric}"] = "N/A"
                continue
                
            # Calculate statistics for each metric
            for metric in metrics:
                if metric in df.columns:
                    values = group_data[metric].dropna()
                    
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        row[f"{alg}_{metric}"] = f"{mean_val:.3f}±{std_val:.3f} (n={len(values)})"
                    else:
                        row[f"{alg}_{metric}"] = "N/A"
                
        summary_rows.append(row)
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_rows)
    output_path = os.path.join(output_dir, "algorithm_comparison_detailed.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"Saved detailed algorithm comparison CSV to {output_path}")
    
    # Also create a simplified version with just the top3_avg metric
    simplified_rows = []
    for dataset in datasets:
        row = {"Dataset": dataset}
        
        for alg in algorithms:
            group_data = df[(df["dataset"] == dataset) & (df["algorithm"] == alg)]
            
            if len(group_data) == 0 or "top3_avg_eval_success_rate" not in df.columns:
                row[f"{alg}_success"] = "N/A"
                continue
                
            values = group_data["top3_avg_eval_success_rate"].dropna()
            
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                row[f"{alg}_success"] = f"{mean_val:.3f}±{std_val:.3f} (n={len(values)})"
            else:
                row[f"{alg}_success"] = "N/A"
                
        simplified_rows.append(row)
    
    # Create simplified summary DataFrame and save to CSV
    simplified_df = pd.DataFrame(simplified_rows)
    simplified_path = os.path.join(output_dir, "algorithm_comparison_summary.csv")
    simplified_df.to_csv(simplified_path, index=False)
    print(f"Saved simplified algorithm comparison summary CSV to {simplified_path}")
    
    return output_path

def plot_reward_model_comparisons(df, output_dir="reward_model_plots"):
    """Create plots comparing algorithm performance across datasets and reward models.
    
    Args:
        df: DataFrame with run information
        output_dir: Directory to save plots
        
    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out rows with missing data
    df_filtered = df.dropna(subset=["top3_avg_eval_success_rate"])
    
    # Process reward model paths for better display
    if "reward_model" in df_filtered.columns:
        # Extract just the relevant parts of the reward model path for display
        def extract_reward_model_name(path):
            if not isinstance(path, str) or path == "none":
                return "none"
            
            # Try to extract informative parts of the path
            path_obj = Path(path)
            
            # Extract the parent directory name
            parent_dir = path_obj.parent.name
            
            # Extract the part after "active_" and include "_aug" if present
            if "active_" in parent_dir:
                parts = parent_dir.split("active_")
                if len(parts) > 1:
                    method = parts[1].split("_")[0]  # Get the first part after active_
                    
                    # Check if "_aug" is at the end
                    if parent_dir.endswith("_aug") or parent_dir.endswith("_augTrue"):
                        method += "_aug"
                    
                    return method
            
            # If we can't extract the method, return the full parent directory
            return parent_dir
        
        df_filtered["reward_model_name"] = df_filtered["reward_model"].apply(extract_reward_model_name)
        
        # Extract number of queries from reward model path
        def extract_num_queries(path):
            if not isinstance(path, str) or path == "none":
                return None
            
            # Look for "max" followed by a number in the path
            match = re.search(r'max(\d+)', path)
            if match:
                return int(match.group(1))
            return None
        
        df_filtered["num_queries"] = df_filtered["reward_model"].apply(extract_num_queries)
        print(f"Extracted num_queries from {df_filtered['num_queries'].notna().sum()} reward models")
    else:
        # If reward_model column doesn't exist, create dummy ones
        df_filtered["reward_model_name"] = "unknown"
        df_filtered["num_queries"] = None
    
    if len(df_filtered) == 0:
        print("No runs with valid success rate metrics found. Cannot generate plots.")
        return output_dir
    
    print(f"Using {len(df_filtered)} runs with valid metrics for reward model comparison plots")
    
    # Metrics to plot
    metrics_to_plot = [
        {"column": "top3_avg_eval_success_rate", "title": "Average of Top 3 Checkpoints Success Rate"}
    ]
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Get unique datasets and algorithms
    datasets = df_filtered["dataset"].unique()
    algorithms = df_filtered["algorithm"].unique()
    
    print(f"Found {len(datasets)} datasets to plot: {', '.join(datasets)}")
    print(f"Found {len(algorithms)} algorithms to plot: {', '.join(algorithms)}")
    # For each dataset and algorithm combination, create plots comparing reward models
    for dataset in datasets:
        for algorithm in algorithms:
            # Filter data for this dataset and algorithm
            dataset_algo_df = df_filtered[(df_filtered["dataset"] == dataset) & 
                                         (df_filtered["algorithm"] == algorithm)]
            
            # Handle zero_rewards_iql case - if algorithm is iql and num_queries is NaN, it's zero_rewards_iql
            if algorithm == "iql":
                dataset_algo_df.loc[dataset_algo_df["num_queries"].isna(), "num_queries"] = 0
                dataset_algo_df.loc[dataset_algo_df["num_queries"] == 0, "reward_model_name"] = "zero_rewards_iql"
            
            # Plot BC as a baseline
            dataset_bc_df = df_filtered[(df_filtered["dataset"] == dataset) & 
                                         (df_filtered["algorithm"] == "bc")]
            
            # Skip if no data for this combination
            if len(dataset_algo_df) == 0:
                continue
            
            # Get unique query counts for this dataset/algorithm combination
            # For IQL, exclude 0 from creating its own plot but include it in other plots
            if algorithm == "iql":
                query_counts = [q for q in dataset_algo_df["num_queries"].unique() if q != 0]
            else:
                query_counts = dataset_algo_df["num_queries"].unique()
            
            # If no query counts found, create a single plot without query count separation
            if len(query_counts) == 0:
                query_counts = [None]
                
            print(f"Found {len(query_counts)} unique query counts for {algorithm} on {dataset}")
            
            # For each query count, create a separate plot
            for num_queries in query_counts:
                # Filter by query count if it's not None
                if num_queries is not None:
                    if algorithm == "iql":
                        # For IQL, include zero query runs in each plot
                        current_df = dataset_algo_df[(dataset_algo_df["num_queries"] == num_queries) | 
                                                    (dataset_algo_df["num_queries"] == 0)]
                    else:
                        current_df = dataset_algo_df[dataset_algo_df["num_queries"] == num_queries]
                    query_label = f"Queries: {num_queries}"
                else:
                    current_df = dataset_algo_df
                    query_label = "Unknown query count"
                
                # Group by reward model
                reward_models = current_df["reward_model_name"].unique()
                
                # Skip if no reward model
                if len(reward_models) < 1:
                    print(f"Skipping {algorithm} on {dataset} with {query_label} - no reward model: {reward_models}")
                    continue
                    
                print(f"Creating plots for {algorithm} on {dataset} with {query_label} ({len(reward_models)} reward models)")
                
                # For each metric, create a plot
                for metric in metrics_to_plot:
                    metric_column = metric["column"]
                    metric_title = metric["title"]
                    
                    # Skip if metric not available
                    if metric_column not in current_df.columns:
                        continue
                    
                    # Group and analyze by reward model
                    grouped = current_df.groupby(["reward_model_name"])
                    
                    # Calculate statistics for the current metric
                    stats = grouped[metric_column].agg([
                        ("mean", np.mean),
                        ("std", np.std),
                        ("min", np.min),
                        ("max", np.max),
                        ("count", "count"),
                        ("median", np.median)
                    ]).reset_index()
                    
                    # Add 95% confidence interval
                    stats["ci_95"] = 1.96 * stats["std"] / np.sqrt(stats["count"].clip(1))
                    
                    # Round statistics for readability
                    for col in ["mean", "std", "min", "max", "median", "ci_95"]:
                        stats[col] = stats[col].round(3)
                    
                    # Sort by mean performance (descending)
                    stats = stats.sort_values("mean", ascending=False)
                    
                    # Set up the figure - adjust height based on number of reward models for better x-label spacing
                    fig_width = max(12, (len(reward_models) + 1) *1.5)
                    fig_height = 8  # Increased height to accommodate x-labels
                    plt.figure(figsize=(fig_width, fig_height))
                    
                    # Create bar plot with error bars
                    ax = sns.barplot(
                        x="reward_model_name", 
                        y="mean", 
                        data=stats,
                        palette="viridis",
                        alpha=0.8
                    )
                    
                    # Add error bars for standard deviation
                    for i, row in enumerate(stats.itertuples()):
                        plt.errorbar(
                            i, row.mean, 
                            yerr=row.std, 
                            fmt='none', 
                            ecolor='black', 
                            capsize=5
                        )
                    
                    # Add data labels on top of bars
                    for i, row in enumerate(stats.itertuples()):
                        plt.text(
                            i, row.mean + 0.02, 
                            f"{row.mean:.3f}±{row.std:.3f}\nn={row.count}", 
                            ha='center', 
                            va='bottom',
                            fontsize=12
                        )
                    
                    # Add BC baseline if available
                    if len(dataset_bc_df) > 0 and metric_column in dataset_bc_df.columns:
                        bc_values = dataset_bc_df[metric_column].dropna()
                        if len(bc_values) > 0:
                            bc_mean = bc_values.mean()
                            bc_std = bc_values.std()
                            bc_count = len(bc_values)
                            
                            # Add horizontal line for BC baseline
                            plt.axhline(y=bc_mean, color='r', linestyle='--', alpha=0.7, 
                                       label=f"BC: {bc_mean:.3f}±{bc_std:.3f} (n={bc_count})")
                            
                            # Add shaded area for BC standard deviation
                            plt.fill_between(
                                [-0.5, len(stats) - 0.5], 
                                bc_mean - bc_std, 
                                bc_mean + bc_std, 
                                color='r', 
                                alpha=0.1
                            )
                            
                            # Add legend
                            plt.legend(loc='upper right', fontsize=12)
                    
                    # Add a title and labels
                    plt.title(f"Reward Model Comparison: {algorithm} on {dataset}\nQueries: {int(num_queries) if num_queries is not None else 'Unknown'}", fontsize=12)
                    plt.ylabel("Success Rate")
                    plt.xlabel("Reward Model")
                    
                    # Rotate x-axis labels for better readability with long reward model names
                    plt.xticks(rotation=15, ha='right')
                    plt.subplots_adjust(bottom=0.3)
                    
                    # Add gridlines for readability
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Adjust the y-axis to start at 0 and have some headroom
                    max_value = stats["mean"].max() + stats["std"].max()
                    plt.ylim(0, min(1.0, max_value * 1.2))
                    
                    # Save the figure with tight layout and extra bottom padding
                    safe_dataset = str(dataset).replace("/", "_")
                    safe_algorithm = str(algorithm).replace("/", "_")
                    plt.tight_layout(pad=2.0, rect=[0, 0.15, 1, 0.95])  # Add extra padding at bottom
                    output_path = os.path.join(output_dir, f"{safe_dataset}_{safe_algorithm}_queries{int(num_queries) if num_queries is not None else 'Unknown'}_reward_model_comparison.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    print(f"Saved reward model comparison plot to {output_path}")
                    plt.close()
    
    # Also create a summary CSV with reward model performance
    create_reward_model_summary_csv(df_filtered, output_dir)
    
    return output_dir

def create_reward_model_summary_csv(df, output_dir="reward_model_plots"):
    """Create a summary CSV with reward model performance by dataset and algorithm."""
    # Create a CSV summary of reward model performance
    summary_rows = []
    
    # Process reward model paths for better display if needed
    if "reward_model_name" not in df.columns and "reward_model" in df.columns:
        # Extract just the relevant parts of the reward model path for display
        def extract_reward_model_name(path):
            if not isinstance(path, str) or path == "none":
                return "none"
            
            # Try to extract informative parts of the path
            path_obj = Path(path)
            
            # Look for directories that might contain model parameters
            # Check if parent directory contains model parameters
            parent_dir = path_obj.parent.name
            if any(param in parent_dir for param in ["n", "k", "seed", "dtw", "model", "seg", "hidden", "epochs"]):
                return parent_dir
            
            # Fall back to filename if parent directory not informative
            return path_obj.name
        
        df["reward_model_name"] = df["reward_model"].apply(extract_reward_model_name)
    
    # Group by dataset, algorithm, and reward model
    if all(col in df.columns for col in ["dataset", "algorithm", "reward_model_name"]):
        grouped = df.groupby(["dataset", "algorithm", "reward_model_name"])
        
        for (dataset, algorithm, reward_model), group_data in grouped:
            row = {
                "Dataset": dataset,
                "Algorithm": algorithm,
                "Reward Model": reward_model
            }
            
            # Add success rate metrics
            if "top3_avg_eval_success_rate" in df.columns:
                values = group_data["top3_avg_eval_success_rate"].dropna()
                
                if len(values) > 0:
                    row["Mean"] = values.mean()
                    row["Std"] = values.std()
                    row["Min"] = values.min()
                    row["Max"] = values.max()
                    row["Count"] = len(values)
                    
                    # Format for display
                    row["Success Rate"] = f"{row['Mean']:.3f}±{row['Std']:.3f} (n={row['Count']})"
                else:
                    row["Success Rate"] = "N/A"
            
            summary_rows.append(row)
    
    # Create summary DataFrame and save to CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        
        # Sort by dataset, algorithm, and mean success rate (descending)
        if "Mean" in summary_df.columns:
            summary_df = summary_df.sort_values(["Dataset", "Algorithm", "Mean"], 
                                              ascending=[True, True, False])
        
        output_path = os.path.join(output_dir, "reward_model_comparison.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"Saved reward model comparison CSV to {output_path}")
        
        # Create a pivoted version for easier reading
        if "Success Rate" in summary_df.columns:
            try:
                pivot_df = summary_df.pivot(index=["Dataset", "Algorithm"], 
                                          columns="Reward Model", 
                                          values="Success Rate")
                pivot_path = os.path.join(output_dir, "reward_model_comparison_pivot.csv")
                pivot_df.to_csv(pivot_path)
                print(f"Saved pivoted reward model comparison CSV to {pivot_path}")
            except Exception as e:
                print(f"Could not create pivoted CSV: {e}")
    
    return output_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and analyze wandb runs for IQL policy")
    parser.add_argument("--project", type=str, default="robot_pref", help="wandb project name")
    parser.add_argument("--entity", type=str, default="clvr", help="wandb entity name")
    parser.add_argument("--user", type=str, default="aliangdw", help="Filter for runs by this user")
    parser.add_argument("--max_runs", type=int, default=None, help="Maximum number of runs to fetch")
    parser.add_argument("--output_dir", type=str, default="analysis", help="Directory to save analysis outputs")
    parser.add_argument("--filter", type=str, default=None, help="JSON string with additional filters")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--plot_dir", type=str, default="algorithm_plots", help="Directory to save algorithm comparison plots")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse additional filters if provided
    filters = None
    if args.filter:
        try:
            filters = json.loads(args.filter)
        except json.JSONDecodeError:
            print(f"Error: Could not parse filter JSON: {args.filter}")
            return
    else:
        # Create default filters
        filters = {}
    
    # Add user filter if specified
    if args.user:
        filters["user"] = args.user
        print(f"Filtering for runs by user: {args.user}")
    
    # Fetch runs
    run_data = fetch_wandb_runs(
        project=args.project,
        entity=args.entity,
        filters=filters,
        max_runs=args.max_runs
    )
    if not run_data:
        print("No runs found matching the criteria.")
        return
    
    # Convert to DataFrame
    df = create_run_dataframe(run_data)
        
    # Print summary of runs with valid metrics
    print("\nSummary of runs with valid metrics:")
    valid_metrics_count = len(df[df["top3_avg_eval_success_rate"].notna()])
    total_runs = len(df)
    print(f"Total runs: {total_runs}, Runs with valid metrics: {valid_metrics_count} ({valid_metrics_count/total_runs:.1%})")
    
    # Group by algorithm and dataset
    if "algorithm" in df.columns and "dataset" in df.columns:
        grouped = df.groupby(["algorithm", "dataset"])
        print("\nMetrics by algorithm and dataset:")
        
        # For each group, count the number of runs with valid metrics
        for (alg, dataset), group in grouped:
            valid_count = len(group[group["top3_avg_eval_success_rate"].notna()])
            total_count = len(group)
            if valid_count > 0:
                avg_success = group["top3_avg_eval_success_rate"].mean()
                print(f"{alg} on {dataset}: {valid_count}/{total_count} runs with valid metrics, avg: {avg_success:.4f}")
            else:
                print(f"{alg} on {dataset}: {valid_count}/{total_count} runs with valid metrics")
    
    # Save to CSV
    csv_path = os.path.join(args.output_dir, "runs_summary.csv")
    top3_csv_path = save_summary_csv(df, csv_path)
    
    # Generate algorithm comparison plots if requested
    if args.plot:
        print("\nGenerating algorithm comparison plots...")
        plot_output_dir = os.path.join(args.output_dir, args.plot_dir)
        plot_algorithm_comparisons(df, plot_output_dir)
        print(f"Plots saved to {plot_output_dir}")
        
        # Generate reward model comparison plots
        print("\nGenerating reward model comparison plots...")
        reward_model_plot_dir = os.path.join(args.output_dir, "reward_model_plots")
        plot_reward_model_comparisons(df, reward_model_plot_dir)
        print(f"Reward model comparison plots saved to {reward_model_plot_dir}")
    
    print(f"\nAnalysis complete. Data saved to {csv_path}")

if __name__ == "__main__":
    main() 