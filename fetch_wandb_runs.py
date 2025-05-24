#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

import wandb

sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
plt.rc("text", usetex=True)  # camera-ready formatting + latex in plots


def create_display_name(path):
    if path == "zero_rewards" or path == "ground_truth":
        return path
    
    if "state_action_reward_model" in path:
        return path

    try:
        # Extract parent directory which contains config info
        path_obj = Path(path)
        # In case we are using intermittent saved checkpoints
        parent_dir = path_obj.parent.name if "checkpoints" not in path_obj.parent.name else path_obj.parent.parent.name
        # Extract key components from the path
        components = {}

        # Look for segment length
        seg_match = re.search(r"seg(\d+)", parent_dir)
        if seg_match:
            components["seg"] = f"S{seg_match.group(1)}"

        # Look for epoch count
        epochs_match = re.search(r"epochs(\d+)", parent_dir)
        if epochs_match:
            components["epochs"] = f"E{epochs_match.group(1)}"

        # Look for pairs/queries count
        pairs_match = re.search(r"pairs(\d+)", parent_dir)
        if pairs_match:
            components["pairs"] = f"Q{pairs_match.group(1)}"

        # Look for iteration (for active learning)
        iter_match = re.search(r"iter[_]?(\d+)", path_obj.name)
        if iter_match:
            components["iter"] = f"I{iter_match.group(1)}"

        # Look for active learning method
        if "active_" in parent_dir:
            method_match = re.search(r"active_([a-zA-Z]+)", parent_dir)
            if method_match:
                components["method"] = method_match.group(1)

        aug_match = re.search(r"aug([a-zA-Z]+)", parent_dir)
        if aug_match:
            match = aug_match.group(1) 
            if match == "True":
                components["aug"] = "AG"

        beta_match = re.search(r"beta([a-zA-Z]+)", parent_dir)
        if beta_match:
            match = beta_match.group(1)
            if match == "True":
                components["beta"] = "BH" # Beta heuristic

        max_match = re.search(r"max(\d+)", parent_dir)
        if max_match:
            components["total_queries"] = f"TQ{max_match.group(1)}"

        k_match = re.search(r"k(\d+)", parent_dir)
        if k_match:
            components["num_augment"] = f"k{k_match.group(1)}"

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
    print("Saved run data to run_data.csv")
    return df


def extract_iter_from_path(path):
    """Extracts the iteration number from a reward model path, if present. Returns None if not found."""
    if not isinstance(path, str):
        return None
    # Try to find iter or iter_ followed by digits
    match = re.search(r"iter[_]?(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def plot_reward_model_comparisons(df, output_dir="reward_model_plots"):
    """Create plots comparing algorithm performance across datasets and reward models.
    Plots are grouped by unique 'iter' key instances if present in reward model path.

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
            (df_filtered["data.data_path"] == dataset)
            & (df_filtered["algorithm"] == "iql")
        ].copy()

        # Add 'iter' column based on reward model path
        filtered_df["iter"] = filtered_df["data.reward_model_path"].apply(extract_iter_from_path)

        # Create a cleaner key from the model path
        filtered_df.loc[:, "key"] = filtered_df["data.reward_model_path"].apply(
            lambda path: create_display_name(path) if isinstance(path, str) else path
        )

        if len(filtered_df) == 0:
            continue

        # Special case: BC comparison baseline
        bc_df = df_filtered[
            (df_filtered["data.data_path"] == dataset)
            & (df_filtered["algorithm"] == "bc")
        ].copy()

        if len(bc_df) > 0:
            bc_df.loc[:, "key"] = "BC"
            bc_df["iter"] = None  # BC does not have an iter

            # Add BC to filtered_df
            filtered_df = pd.concat([filtered_df, bc_df])

        # Special case: use_zero_rewards
        filtered_df.loc[
            (filtered_df["data.data_path"] == dataset)
            & (filtered_df["algorithm"] == "iql")
            & (filtered_df["data.use_zero_rewards"] == True),
            "key",
        ] = "Zero"

        # Special case: use_ground_truth
        filtered_df.loc[
            (filtered_df["data.data_path"] == dataset)
            & (filtered_df["algorithm"] == "iql")
            & (filtered_df["data.use_ground_truth"] == True),
            "key",
        ] = "GT"

        # Find all unique iter values (including None for non-iter runs)
        unique_iters = sorted(filtered_df["iter"].dropna().unique())
        # If there are no iter values, just plot once (legacy behavior)
        if not unique_iters:
            unique_iters = [None]

        print(f"Unique iter values: {unique_iters}")
        
        # Create a single figure with 3 columns of subplots
        num_iters = len(unique_iters)
        # num_rows = (num_iters + 2) // 3  # Ceiling division to determine rows needed
        num_rows = 2 # TODO: idk why but only if hardcoded to 2, it works
        
        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows), constrained_layout=True)
        
        dataset_name = Path(dataset).name
        # Add a main title for the entire figure
        fig.suptitle(f"Reward Model Comparison on {dataset_name}", fontsize=20, y=1.02)

        for i, iter_value in enumerate(unique_iters):
            # Calculate row and column for current subplot
            row_idx = i // 3
            col_idx = i % 3

            # axes is 2D array of shape (num_rows, 3)
            if row_idx >= axes.shape[0] or col_idx >= axes.shape[1]:
                # No axis for this subplot, skip
                continue
            ax = axes[row_idx, col_idx]

            if iter_value is not None:
                iter_mask = filtered_df["iter"] == iter_value
                plot_df = filtered_df[iter_mask | filtered_df["iter"].isna()].copy()
                iter_str = f"_iter{iter_value}"
                print(f"Plotting for dataset {dataset} at iter={iter_value} with {len(plot_df)} runs")
            else:
                # Plot all runs with iter=None (e.g., BC, GT, Zero, or non-iter reward models)
                plot_df = filtered_df[filtered_df["iter"].isna()].copy()
                iter_str = ""
                print(f"Plotting for dataset {dataset} with no iter (baseline/non-iter) with {len(plot_df)} runs")

            if len(plot_df) == 0:
                # Hide this axis and continue
                ax.set_visible(False)
                continue

            # Calculate statistics for each reward model
            metric_column = "top3_avg_eval_success_rate"
            grouped = plot_df.groupby(["key"])

            stats_dict = {
                "mean": grouped[metric_column].mean(),
                "std": grouped[metric_column].std(),
                "min": grouped[metric_column].min(),
                "max": grouped[metric_column].max(),
                "count": grouped[metric_column].count(),
                "median": grouped[metric_column].median(),
            }

            stats = pd.DataFrame(stats_dict).reset_index()
            stats = stats.sort_values("mean", ascending=True)

            # drop the reward_model/state_action_reward_model.pt
            stats = stats[stats["key"] != "reward_model/state_action_reward_model.pt"]

            # Create the bar plot
            sns.barplot(
                x="key",
                y="mean",
                data=stats,
                palette="viridis",
                ax=ax
            )

            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add error bars and data labels
            for j, row in enumerate(stats.itertuples()):
                ax.errorbar(
                    j, row.mean, yerr=row.std, fmt="none", ecolor="black", capsize=5
                )
                ax.text(
                    j,
                    row.mean + 0.02,
                    f"{row.mean:.3f}±{row.std:.3f}\nn={row.count}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Set subplot title
            if iter_value is not None:
                ax.set_title(f"Iteration {iter_value}", fontsize=14)
            else:
                ax.set_title("No Iteration (Baselines)", fontsize=14)
                
            ax.set_ylabel("Success Rate")
            ax.set_xlabel("Reward Model")
            ax.tick_params(axis='x', labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=45)

            # Set y-axis limits
            max_value = stats["mean"].max() + stats["std"].max()
            ax.set_ylim(0, min(1.0, max_value * 1.2))

        # # Hide any unused subplots
        # for i in range(len(unique_iters), num_rows * 3):
        #     row_idx = i // 3
        #     col_idx = i % 3
        #     if num_rows == 1:
        #         axes[col_idx].set_visible(False)
        #     else:
        #         axes[row_idx, col_idx].set_visible(False)

        # Save the figure
        output_path = os.path.join(
            output_dir,
            f"{dataset_name}_rm_analysis_combined.png",
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined reward model comparison plot to {output_path}")
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
