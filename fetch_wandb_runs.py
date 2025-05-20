#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import hydra
from omegaconf import DictConfig, OmegaConf

import wandb

sns.set_style("white")
sns.set_style("ticks")
sns.set_context('talk')
plt.rc('text', usetex=True)  # camera-ready formatting + latex in plots

def create_display_name(path):
    if path == "zero_rewards" or path == "ground_truth":
        return path
        
    try:
        # Extract parent directory which contains config info
        path_obj = Path(path)
        parent_dir = path_obj.parent.name
        
        # Extract key components from the path
        components = {}
        
        # Look for segment length
        seg_match = re.search(r'seg(\d+)', parent_dir)
        if seg_match:
            components["seg"] = f"S{seg_match.group(1)}"
        
        # Look for epoch count
        epochs_match = re.search(r'epochs(\d+)', parent_dir)
        if epochs_match:
            components["epochs"] = f"E{epochs_match.group(1)}"
        
        # Look for pairs/queries count
        pairs_match = re.search(r'pairs(\d+)', parent_dir)
        if pairs_match:
            components["pairs"] = f"Q{pairs_match.group(1)}"
        
        # Look for iteration (for active learning)
        iter_match = re.search(r'iter(\d+)', path_obj.name)
        if iter_match:
            components["iter"] = f"I{iter_match.group(1)}"
        
        # Look for active learning method
        if "active_" in parent_dir:
            method_match = re.search(r'active_(\w+)', parent_dir)
            if method_match:
                components["method"] = method_match.group(1)
        
        # Build the display string
        if components:
            # Sort components by key to ensure consistent ordering
            display_parts = [v for k, v in sorted(components.items())]
            return "-".join(display_parts)
        else:
            # Fallback to filename if no components extracted
            return path_obj.name
    except Exception:
        # Fallback for any errors
        return str(path).split("/")[-1]
            
def fetch_wandb_runs(project="robot_pref", entity="clvr", filters=None, max_runs=None, after_date=None):
    """Fetch runs from wandb.

    Args:
        project: Name of the wandb project
        entity: Name of the wandb entity
        filters: Dictionary of filters to apply to the runs
        max_runs: Maximum number of runs to fetch
        after_date: Only include runs created after this date (format: YYYY-MM-DD)

    Returns:
        List of run data dictionaries
    """
    print(f"Fetching runs from wandb project: {entity}/{project}")
    api = wandb.Api()
    user_filter = filters.get("user") if filters else None
    if user_filter:
        print(f"Will filter for user: {user_filter}")
    
    # Parse date filter if provided
    date_filter = None
    if after_date:
        try:
            date_filter = pd.to_datetime(after_date)
            print(f"Will filter for runs created after: {after_date}")
        except:
            print(f"Warning: Invalid date format '{after_date}'. Expected format: YYYY-MM-DD")

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
        
        # Apply date filter if specified
        if date_filter and include_run:
            # Convert run_date to naive datetime by removing timezone info to match date_filter
            run_date = pd.to_datetime(run.created_at)
            if run_date.tzinfo is not None:
                run_date = run_date.replace(tzinfo=None)
            if run_date < date_filter:
                include_run = False

        if include_run:
            filtered_runs.append(run)

    print(f"After filtering: {len(filtered_runs)} runs")

    if max_runs is not None and len(filtered_runs) > max_runs:
        filtered_runs = filtered_runs[:max_runs]
        print(f"Limited to {len(filtered_runs)} runs")

    run_data = []
    for run in filtered_runs:
        run_dict = {
            "id": run.id,
            "name": run.name,
            "tags": run.tags,
            "state": run.state,
            "created_at": run.created_at,
            "config": run.config,
            "summary": run.summary._json_dict,
            "url": run.url,
        }
        if hasattr(run, "user"):
            run_dict["user"] = run.user.username

        # Try both possible epoch keys
        history = run.history(keys=["eval/success_rate", "eval/epoch"])
        if history.empty:
            history = run.history(keys=["eval/success_rate", "epoch"])

        if not history.empty and "eval/success_rate" in history.columns:
            success_rates = history["eval/success_rate"].dropna().tolist()
            epochs = (
                history["epoch"].dropna().tolist()
                if "epoch" in history.columns
                else list(range(len(success_rates)))
            )
            run_dict["history"] = {
                "eval_success_rates": success_rates,
                "epochs": epochs,
            }
            if success_rates:
                print(f"Run {run.name}: Found {len(success_rates)} success rate values")
                run_data.append(run_dict)
            else:
                print(f"Run {run.name}: No success rate values found")
        else:
            print(f"Run {run.name}: No eval/success_rate in history")

    print(f"Extracted data from {len(run_data)} runs with success rate metrics")
    return run_data


def save_run_df(run_data, output_dir="run_data"):
    """Convert run data to a pandas DataFrame with relevant metrics."""
    rows = []
    for run in run_data:
        # Basic run information
        row = {
            "run_id": run["id"],
            "run_name": run["name"],
            "state": run["state"],
            "created_at": run["created_at"],
            "url": run["url"],
            "user": run.get("user", "unknown"),
        }
        
        # Extract config information
        config = run.get("config", {})
        
        # Process top-level config fields
        for k, v in config.items():
            if not isinstance(v, dict):
                row[k] = v
        
        # Process nested config fields
        for section_key, section_data in config.items():
            if isinstance(section_data, dict):
                for k, v in section_data.items():
                    # Use section_key.k naming convention for nested fields
                    field_name = f"{section_key}.{k}"
                    row[field_name] = v

        # Add history data for success rates
        if "history" in run and "eval_success_rates" in run["history"]:
            success_rates = run["history"]["eval_success_rates"]

            # Get the latest success rate
            row["latest_eval_success_rate"] = success_rates[-1]
            row["max_eval_success_rate"] = max(success_rates)
            top_3_rates = sorted(success_rates, reverse=True)[:3]
            row["top3_avg_eval_success_rate"] = sum(top_3_rates) / len(top_3_rates)
            for i, rate in enumerate(top_3_rates):
                row[f"top_eval_success_{i + 1}"] = rate
            row["history_eval_success_rates"] = success_rates
            row["history_epochs"] = run["history"].get("epochs", [])
        
        rows.append(row)

    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at", ascending=False)

    print("\nSummary of runs with valid metrics:")
    valid_metrics_count = len(df[df["top3_avg_eval_success_rate"].notna()])
    total_runs = len(df)
    print(
        f"Total runs: {total_runs}, Runs with valid metrics: {valid_metrics_count} ({valid_metrics_count / total_runs:.1%})"
    )
    df.to_csv(os.path.join(output_dir, "run_data.csv"), index=False)
    print(f"Saved run data to run_data.csv")
    return df

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
    
    if len(df_filtered) == 0:
        print("No runs with valid success rate metrics found. Cannot generate plots.")
        return output_dir
            
    # Get unique datasets and algorithms
    datasets = df_filtered["data.data_path"].unique()
    algorithms = df_filtered["algorithm"].unique()
    print(f"Found {len(datasets)} datasets to plot: {datasets}")
    print(f"Found {len(algorithms)} algorithms to plot: {algorithms}")
    
    # Loop through each dataset (each separate plot)
    for dataset in datasets:
        # Filter data for this dataset and algorithm
        filtered_df = df_filtered[
            (df_filtered["data.data_path"] == dataset) & 
            (df_filtered["algorithm"] == "iql")
        ]

        # Create a cleaner key from the model path
        filtered_df.loc[:, "key"] = filtered_df["data.reward_model_path"].apply(
            lambda path: create_display_name(path) if isinstance(path, str) else path
        )
        
        if len(filtered_df) == 0:
            continue
                        
        # Special case: BC comparison baseline
        bc_df = df_filtered[
            (df_filtered["data.data_path"] == dataset) & 
            (df_filtered["algorithm"] == "bc")
        ]
        bc_df.loc[:, "key"] = "BC"

        # Add BC to filtered_df
        filtered_df = pd.concat([filtered_df, bc_df])

        # Special case: use_zero_rewards 
        filtered_df.loc[
            (filtered_df["data.data_path"] == dataset) & 
            (filtered_df["algorithm"] == "iql") &
            (filtered_df["data.use_zero_rewards"] == True),
            "key"
        ] = "Zero"

        # Special case: use_ground_truth
        filtered_df.loc[
            (filtered_df["data.data_path"] == dataset) & 
            (filtered_df["algorithm"] == "iql") &
            (filtered_df["data.use_ground_truth"] == True),
            "key"
        ] = "GT"

         # Set up figure and colors
        plt.figure(figsize=(8, 6))        
    
        # Calculate statistics for each reward model
        metric_column = "top3_avg_eval_success_rate"
        grouped = filtered_df.groupby(["key"])
        
        stats_dict = {
            "mean": grouped[metric_column].mean(),
            "std": grouped[metric_column].std(),
            "min": grouped[metric_column].min(),
            "max": grouped[metric_column].max(),
            "count": grouped[metric_column].count(),
            "median": grouped[metric_column].median()
        }
        
        stats = pd.DataFrame(stats_dict).reset_index()
        stats = stats.sort_values("mean", ascending=True)

        # drop the reward_model/state_action_reward_model.pt
        stats = stats[stats["key"] != "reward_model/state_action_reward_model.pt"]
        
        # Create the bar plot
        ax = sns.barplot(
            x="key", 
            y="mean", 
            data=stats,
            palette="viridis",
        )
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add error bars and data labels
        for i, row in enumerate(stats.itertuples()):
            plt.errorbar(i, row.mean, yerr=row.std, fmt="none", ecolor="black", capsize=5)
            plt.text(
                i, row.mean + 0.02, 
                f"{row.mean:.3f}±{row.std:.3f}\nn={row.count}",
                ha="center", va="bottom", fontsize=10,
            )
            
        # Add title and labels
        dataset_name = Path(dataset).name
        plt.title(
            f"Reward Model Comparison on {dataset_name}",
            fontsize=18,
        )
        plt.ylabel("Success Rate")
        plt.xlabel("Reward Model")
        plt.xticks(rotation=45, ha="right")
        
        # Format and save the plot
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        max_value = stats["mean"].max() + stats["std"].max()
        plt.ylim(0, min(1.0, max_value * 1.2))
                        
        output_path = os.path.join(
            output_dir,
            f"{dataset_name}_rm_analysis.png",
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved reward model comparison plot to {output_path}")
        plt.close()

    return output_dir


@hydra.main(config_path="config", config_name="fetch_wandb", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function to fetch and analyze wandb runs."""
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))
    
    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    
    # Parse filters
    filters = {}
    if cfg.wandb.filter:
        try:
            filters = json.loads(cfg.wandb.filter)
        except json.JSONDecodeError:
            print(f"Error: Could not parse filter JSON: {cfg.wandb.filter}")
            return
    
    if cfg.wandb.user:
        filters["user"] = cfg.wandb.user
        print(f"Filtering for runs by user: {cfg.wandb.user}")
    
    print(f"Filters: {filters}")

    # Fetch run data
    run_data = fetch_wandb_runs(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        filters=filters,
        max_runs=cfg.wandb.max_runs,
        after_date=cfg.wandb.after_date,
    )
    
    if not run_data:
        print("No runs found matching the criteria.")
        return

    # Process data
    df = save_run_df(run_data, cfg.output.output_dir)

    # Generate plots 
    print("\nGenerating reward model comparison plots...")
    plot_reward_model_comparisons(df, cfg.output.output_dir)
    print(f"Reward model comparison plots saved to {cfg.output.output_dir}")


if __name__ == "__main__":
    main()