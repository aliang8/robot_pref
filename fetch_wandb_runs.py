#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

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
                row["dataset"] = Path(config["data"]["data_path"]).stem
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

def group_and_analyze(df):
    """Group data by algorithm and dataset and calculate statistics.
    
    Args:
        df: DataFrame with run information
        
    Returns:
        DataFrame with statistics for each algorithm/dataset combination
    """
    # Ensure algorithm column exists
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
    
    # Group by dataset and algorithm
    grouped = df.groupby(["dataset", "algorithm"])

    # Calculate statistics for each group
    stats = grouped["top3_avg_eval_success_rate"].agg([
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
    
    return stats

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
    
    # Group and analyze data
    stats = group_and_analyze(df_filtered)
    
    # Check if we have any data to plot
    if len(stats) == 0:
        print("No data to plot after grouping. Cannot generate plots.")
        return output_dir
    
    # Get unique datasets and algorithms
    datasets = stats["dataset"].unique()
    algorithms = stats["algorithm"].unique()
    
    print(f"Found {len(datasets)} datasets and {len(algorithms)} algorithms to plot")
    
    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create a plot for each dataset
    for dataset in datasets:
        dataset_stats = stats[stats["dataset"] == dataset]
        
        # Skip if we only have one algorithm
        if len(dataset_stats) <= 1:
            print(f"Skipping {dataset} - only has {len(dataset_stats)} algorithm")
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
        plt.title(f"Algorithm Performance on {dataset} Dataset\n(Average of Top 3 Checkpoints Success Rate)")
        plt.ylabel("Success Rate")
        plt.xlabel("Algorithm")
        
        # Add gridlines for readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust the y-axis to start at 0 and have some headroom
        max_value = dataset_stats["mean"].max() + dataset_stats["std"].max()
        plt.ylim(0, min(1.0, max_value * 1.2))
        
        # Save the figure
        safe_dataset = str(dataset).replace("/", "_")
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{safe_dataset}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()
    
    # Create a summary CSV with algorithm performance by dataset
    create_algorithm_summary_csv(stats, output_dir)
    
    return output_dir

def create_algorithm_summary_csv(stats, output_dir="algorithm_plots"):
    """Create a summary CSV with algorithm performance by dataset."""
    # Reshape the data for better readability
    pivot_mean = stats.pivot(index="dataset", columns="algorithm", values="mean")
    pivot_std = stats.pivot(index="dataset", columns="algorithm", values="std")
    pivot_count = stats.pivot(index="dataset", columns="algorithm", values="count")
    
    # Create a clean CSV with formatted values
    summary_rows = []
    
    for dataset in pivot_mean.index:
        row = {"Dataset": dataset}
        
        for alg in pivot_mean.columns:
            mean_val = pivot_mean.loc[dataset, alg]
            std_val = pivot_std.loc[dataset, alg]
            count_val = pivot_count.loc[dataset, alg]
            
            if pd.notna(mean_val):
                row[f"{alg}_success"] = f"{mean_val:.3f}±{std_val:.3f} (n={int(count_val)})"
            else:
                row[f"{alg}_success"] = "N/A"
                
        summary_rows.append(row)
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_rows)
    output_path = os.path.join(output_dir, "algorithm_comparison_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"Saved algorithm comparison summary CSV to {output_path}")
    
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and analyze wandb runs for IQL policy")
    parser.add_argument("--project", type=str, default="robot_pref", help="wandb project name")
    parser.add_argument("--entity", type=str, default="clvr", help="wandb entity name")
    parser.add_argument("--user", type=str, default="aliangdw", help="Filter for runs by this user")
    parser.add_argument("--max_runs", type=int, default=None, help="Maximum number of runs to fetch")
    parser.add_argument("--output_dir", type=str, default="analysis", help="Directory to save analysis outputs")
    parser.add_argument("--filter", type=str, default=None, help="JSON string with additional filters")
    parser.add_argument("--plot", action="store_true", help="Generate algorithm comparison plots")
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
    csv_path = os.path.join(args.output_dir, f"runs_summary.csv")
    top3_csv_path = save_summary_csv(df, csv_path)
    
    # Generate algorithm comparison plots if requested
    if args.plot:
        print("\nGenerating algorithm comparison plots...")
        plot_output_dir = os.path.join(args.output_dir, args.plot_dir)
        plot_algorithm_comparisons(df, plot_output_dir)
        print(f"Plots saved to {plot_output_dir}")
    
    print(f"\nAnalysis complete. Data saved to {csv_path}")

if __name__ == "__main__":
    main() 