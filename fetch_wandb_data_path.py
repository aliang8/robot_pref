#!/usr/bin/env python3
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from wandb.errors import CommError

sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
plt.rc("text", usetex=True)  # camera-ready formatting + latex in plots

def fetch_wandb_runs(
    project="robot_pref", entity="clvr", filters=None, max_runs=None, after_date=None
):
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
            print(
                f"Warning: Invalid date format '{after_date}'. Expected format: YYYY-MM-DD"
            )

    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} total runs")

    # Filter runs manually
    filtered_runs = []
    for run in tqdm(runs):
        include_run = True

        # Apply user filter if specified
        if user_filter:
            if hasattr(run, "user") and run.user.username != user_filter:
                include_run = False

        # Apply date filter if specified
        if date_filter and include_run:
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

        history = None
        try:
            history = run.history(keys=["eval/eval_success"])
        except CommError:
            print(f"Run {run.name}")
            history = None

        if not history.empty and "eval/eval_success" in history.columns:
            success_rates = [rate/100 for rate in history["eval/eval_success"].dropna().tolist()]
            run_dict["history"] = {
                "eval_success_rates": success_rates,
            }
            if success_rates:
                print(f"Run {run.name}: Found {len(success_rates)} success rate values")
                run_data.append(run_dict)
            else:
                print(f"Run {run.name}: No success rate values found")
        else:
            print(f"Run {run.name}: No eval/eval_success in history")

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
                    field_name = f"{section_key}.{k}"
                    row[field_name] = v

        # Get SR metrics
        if "history" in run and "eval_success_rates" in run["history"]:
            success_rates = run["history"]["eval_success_rates"]
            # top_3_rates = success_rates[-10:]
            top_3_rates = sorted(success_rates, reverse=True)[:3]
            row["top3_avg_eval_success_rate"] = sum(top_3_rates) / len(top_3_rates)

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
    print("Saved run data to run_data.csv")
    return df

def plot_data_path_comparisons(df, output_dir="data_path_plots"):
    """Create plots comparing performance across different data paths and trivial reward settings."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter out rows with missing data
    df_filtered = df.dropna(subset=["top3_avg_eval_success_rate"])

    if len(df_filtered) == 0:
        print("No runs with valid success rate metrics found. Cannot generate plots.")
        return output_dir

    # Get unique data paths from both "data_path" and "_content.data_path" if present
    data_path_set = set()
    if "data_path" in df_filtered.columns:
        data_path_set.update(df_filtered["data_path"].dropna().unique())
    if "_content.data_path" in df_filtered.columns:
        data_path_set.update(df_filtered["_content.data_path"].dropna().unique())
    if not data_path_set:
        raise KeyError("Neither 'data_path' nor '_content.data_path' found in DataFrame columns.")
    data_paths = list(data_path_set)
    print(f"Found {len(data_paths)} unique data paths: {data_paths}")

    # Helper function to get config value with fallback to _content prefixed path
    def get_config_value(row, config_key, default=None):
        """Get config value, trying direct path first, then _content prefixed path."""
        # Try direct path first
        if config_key in row:
            return row[config_key]
        
        # Try _content prefixed path
        content_key = f"_content.{config_key}"
        if content_key in row:
            return row[content_key]
        
        return default

    # Create a plot for each data path
    for data_path in data_paths:
        # Filter data for this data path, considering both "data_path" and "_content.data_path"
        mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
        if "data_path" in df_filtered.columns:
            mask = mask | (df_filtered["data_path"] == data_path)
        if "_content.data_path" in df_filtered.columns:
            mask = mask | (df_filtered["_content.data_path"] == data_path)
        path_df = df_filtered[mask].copy()
        
        # Create separate groups for reward types
        def get_reward_type(row):
            # If BC in name, always "BC"
            run_name = get_config_value(row, "run_name", "")
            if "bc" in run_name.lower():
                return "BC"
            
            # Helper to add Distributional tag
            def add_tag(base, is_dist, feedback_num=None):
                if is_dist:
                    if feedback_num is not None:
                        return f"{base} + Dist RM (NF={feedback_num})"
                    return f"{base} + Dist RM"
                else:
                    if feedback_num is not None:
                        return f"{base} (NF={feedback_num})"
                    return base

            is_dist = get_config_value(row, "use_distributional_model", False) is True
            feedback_num = get_config_value(row, "feedback_num", False)

            if get_config_value(row, "eef_rm", False) is True:
                return add_tag("EEF RM", is_dist, feedback_num)
            elif get_config_value(row, "use_gt_prefs", False) is True or get_config_value(row, "use_gt_aug_prefs", False) is True:
                return add_tag("GT Prefs", is_dist, feedback_num)
            elif get_config_value(row, "trivial_reward", None) == 1:
                return add_tag("Zero Rewards", is_dist)
            elif get_config_value(row, "trivial_reward", None) == 0:
                return add_tag("Aug Prefs", is_dist, feedback_num)
            else:
                return "IQL_Zero"

        path_df["reward_type"] = path_df.apply(get_reward_type, axis=1)
        
        if len(path_df) == 0:
            print(f"No valid reward types found for data path: {data_path}")
            continue
        
        # Calculate statistics for each group
        metric_column = "top3_avg_eval_success_rate"
        grouped = path_df.groupby(["reward_type"])
        
        stats_dict = {
            "mean": grouped[metric_column].mean(),
            "std": grouped[metric_column].std(),
            "count": grouped[metric_column].count(),
        }
        
        stats = pd.DataFrame(stats_dict).reset_index()
        
        # Create the plot
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(
            x="reward_type",
            y="mean",
            data=stats,
            palette="viridis"
        )
        
        # Add error bars and data labels
        for i, row in enumerate(stats.itertuples()):
            ax.errorbar(
                i, row.mean, yerr=row.std, fmt="none", ecolor="black", capsize=5
            )
            ax.text(
                i,
                row.mean + 0.02,
                f"{row.mean:.3f}±{row.std:.3f}\nn={row.count}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        
        # Customize plot
        plt.title(f"Performance Comparison for {Path(data_path).name}", fontsize=14)
        plt.ylabel("Success Rate")
        
        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Set y-axis limits
        max_value = stats["mean"].max() + stats["std"].max()
        plt.ylim(0, min(1.0, max_value * 1.2))
        
        # Save the plot
        output_path = os.path.join(
            output_dir,
            f"{Path(data_path).name}_comparison.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")
        plt.close()

    return output_dir

@hydra.main(config_path="configs", config_name="fetch_wandb", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function to fetch and analyze wandb runs."""
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))

    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)

    # Parse filters
    filters = {}


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
    print("\nGenerating data path comparison plots...")
    plot_data_path_comparisons(df, os.path.join(cfg.output.output_dir, cfg.output.plot_dir))
    print(f"Data path comparison plots saved to {os.path.join(cfg.output.output_dir, cfg.output.plot_dir)}")

if __name__ == "__main__":
    main() 