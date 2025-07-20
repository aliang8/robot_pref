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

# Configure plotting
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("talk")
plt.rc("text", usetex=True)


def fetch_wandb_runs(project, entity, filters=None, max_runs=None, after_date=None):
    """Fetch and filter runs from wandb project."""
    print(f"Fetching runs from wandb project: {entity}/{project}")
    api = wandb.Api()

    # Parse date filter
    date_filter = None
    if after_date:
        try:
            date_filter = pd.to_datetime(after_date)
            print(f"Filtering for runs created after: {after_date}")
        except:
            print(f"Warning: Invalid date format '{after_date}'. Expected: YYYY-MM-DD")

    # Get all runs
    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} total runs")

    # Filter runs
    filtered_runs = []
    for run in tqdm(runs):
        # User filter
        if filters and filters.get("user"):
            if not hasattr(run, "user") or run.user.username != filters["user"]:
                continue

        # Date filter
        if date_filter:
            run_date = pd.to_datetime(run.created_at)
            if run_date.tzinfo:
                run_date = run_date.replace(tzinfo=None)
            if run_date < date_filter:
                continue

        filtered_runs.append(run)

    print(f"After filtering: {len(filtered_runs)} runs")

    # Limit runs if specified
    if max_runs and len(filtered_runs) > max_runs:
        filtered_runs = filtered_runs[:max_runs]
        print(f"Limited to {len(filtered_runs)} runs")

    # Extract run data
    run_data = []
    for run in filtered_runs:
        try:
            history = run.history(keys=["eval/success"])
        except CommError:
            print(f"Run {run.name}: Communication error")
            continue

        # Check for success rate data
        if history.empty or "eval/success" not in history.columns:
            print(f"Run {run.name}: No eval/success in history")
            continue

        success_rates = history["eval/success"].dropna().tolist()
        if not success_rates:
            print(f"Run {run.name}: No success rate values found")
            continue

        # Store run data
        run_dict = {
            "id": run.id,
            "name": run.name,
            "tags": run.tags,
            "state": run.state,
            "created_at": run.created_at,
            "config": run.config,
            "summary": run.summary._json_dict,
            "url": run.url,
            "user": run.user.username if hasattr(run, "user") else "unknown",
            "eval_success_rates": success_rates,
        }

        print(f"Run {run.name}: Found {len(success_rates)} success rate values")
        run_data.append(run_dict)

    print(f"Extracted data from {len(run_data)} runs with success rate metrics")
    return run_data


def create_dataframe(run_data):
    """Convert run data to pandas DataFrame."""
    rows = []

    for run in run_data:
        row = {
            "run_id": run["id"],
            "run_name": run["name"],
            "state": run["state"],
            "created_at": run["created_at"],
            "url": run["url"],
            "user": run["user"],
        }

        # Flatten config
        config = run.get("config", {})
        for key, value in config.items():
            if isinstance(value, dict):
                # Flatten nested config
                for nested_key, nested_value in value.items():
                    row[f"{key}.{nested_key}"] = nested_value
            else:
                row[key] = value

        # Calculate top 3 average success rate
        success_rates = run["eval_success_rates"]
        # top_rates = success_rates[-5:]  # Last 5 rates
        top_rates = sorted(success_rates, reverse=True)[:3]
        row["top3_avg_eval_success_rate"] = sum(top_rates) / len(top_rates)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by creation date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at", ascending=False)

    # Print summary
    valid_metrics = len(df[df["top3_avg_eval_success_rate"].notna()])
    total_runs = len(df)
    print(
        f"\nSummary: {total_runs} runs, {valid_metrics} with valid metrics ({valid_metrics / total_runs:.1%})"
    )

    return df


def get_config_value(row, key):
    """Get config value with fallback to _content prefixed version."""
    if key in row:
        return row[key]
    content_key = f"_content.{key}"
    if content_key in row:
        return row[content_key]
    return None


def classify_reward_type(row):
    """Classify the reward type based on run configuration."""
    run_name = get_config_value(row, "run_name") or ""

    if "bc" in run_name.lower():
        return "bc"

    if get_config_value(row, "trivial_reward") == 1:
        return "zero"

    if get_config_value(row, "use_reward_model") is False:
        return "gt rewards"

    if get_config_value(row, "use_cross") is True:
        if get_config_value(row, "use_goal_pos") is True:
            return "dtw prefs (2d goal + 3d eef)"
        return "dtw prefs (3d eef)"

    if get_config_value(row, "eef_rm") is True:
        if get_config_value(row, "eef_rm_2d") is True:
            return "eef + 2d goal rm"
        return "eef rm"

    if get_config_value(row, "single_emb") is True:
        return None

    if get_config_value(row, "human") is True:
        return "human prefs"

    return "unknown"


def create_comparison_plots(df, output_dir):
    """Create performance comparison plots for each data path."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter valid data
    df_filtered = df.dropna(subset=["top3_avg_eval_success_rate"])

    if len(df_filtered) == 0:
        print("No runs with valid success rate metrics found.")
        return

    # Get unique data paths
    data_paths = set()
    for col in ["data_path", "_content.data_path"]:
        if col in df_filtered.columns:
            data_paths.update(df_filtered[col].dropna().unique())

    if not data_paths:
        raise KeyError("No data_path columns found in DataFrame")

    print(f"Found {len(data_paths)} unique data paths: {list(data_paths)}")

    # Create plot for each data path
    for data_path in data_paths:
        # Filter data for this path
        mask = pd.Series(False, index=df_filtered.index)
        for col in ["data_path", "_content.data_path"]:
            if col in df_filtered.columns:
                mask |= df_filtered[col] == data_path

        path_df = df_filtered[mask].copy()

        if len(path_df) == 0:
            continue

        # Classify reward types
        path_df["reward_type"] = path_df.apply(classify_reward_type, axis=1)

        # Calculate statistics
        stats = (
            path_df.groupby("reward_type")["top3_avg_eval_success_rate"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Plot in ascending order of mean success rate
        plt.figure(figsize=(8, 4))
        stats = stats.sort_values("mean", ascending=True)
        stats = stats.reset_index(drop=True)  # Reset index to match new order
        performance_order = stats["reward_type"].tolist()
        ax = sns.barplot(
            x="reward_type",
            y="mean",
            data=stats,
            palette="viridis",
            order=performance_order,
        )

        # Add error bars and labels - now indices match bar positions
        for i, row in stats.iterrows():
            ax.errorbar(
                i, row["mean"], yerr=row["std"], fmt="none", ecolor="black", capsize=5
            )
            ax.text(
                i,
                row["mean"] + row["std"] + 0.01,
                f"{row['mean']:.3f}±{row['std']:.3f}\nn={row['count']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Customize plot
        plt.title(f"Performance Comparison for {Path(data_path).name}", fontsize=10)
        plt.ylabel("Success Rate", fontsize=8)
        plt.xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set y-axis limits
        max_value = stats["mean"].max() + stats["std"].max()
        plt.ylim(0, min(1.0, max_value * 1.2))

        # Save plot
        output_path = os.path.join(output_dir, f"{Path(data_path).name}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")
        plt.close()


@hydra.main(config_path="configs", config_name="fetch_wandb", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function to fetch and analyze wandb runs."""
    print(OmegaConf.to_yaml(cfg))

    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)

    # Set up filters
    filters = {}
    if cfg.wandb.user:
        filters["user"] = cfg.wandb.user
        print(f"Filtering for runs by user: {cfg.wandb.user}")

    # Fetch and process data
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

    # Create DataFrame and plots
    df = create_dataframe(run_data)
    print("\nGenerating comparison plots...")
    create_comparison_plots(df, cfg.output.output_dir)
    print(f"Plots saved to {cfg.output.output_dir}")


if __name__ == "__main__":
    main()
