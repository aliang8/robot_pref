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
    """Fetch runs from wandb and extract relevant data."""
    print(f"Fetching runs from wandb project: {entity}/{project}")
    api = wandb.Api()
    user_filter = filters.get("user") if filters else None
    if user_filter:
        print(f"Will filter for user: {user_filter}")

    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} total runs")

    # Manual filtering
    filtered_runs = [
        run for run in runs
        if not user_filter or (hasattr(run, "user") and run.user.username == user_filter)
    ]
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


def create_run_dataframe(run_data):
    """Convert run data to a pandas DataFrame with relevant metrics."""
    rows = []
    for run in run_data:
        row = {
            "run_id": run["id"],
            "run_name": run["name"],
            "state": run["state"],
            "created_at": run["created_at"],
            "url": run["url"],
            "user": run.get("user", "unknown"),
        }
        config = run.get("config", {})
        row["algorithm"] = config.get("algorithm", "unknown")
        if "data" in config and "data_path" in config["data"]:
            path_obj = Path(config["data"]["data_path"])
            row["dataset"] = f"{path_obj.parent.name}_{path_obj.stem}"
        else:
            row["dataset"] = "unknown"
        if "data" in config and "reward_model_path" in config["data"]:
            row["reward_model"] = config["data"].get("reward_model_path", "none")
            row["use_ground_truth"] = config["data"].get("use_ground_truth", False)
            row["scale_rewards"] = config["data"].get("scale_rewards", False)
        if "training" in config:
            row["n_epochs"] = config["training"].get("n_epochs", 0)
        if "model" in config:
            row["actor_lr"] = config["model"].get("actor_learning_rate", 0)
            row["critic_lr"] = config["model"].get("critic_learning_rate", 0)
            row["expectile"] = config["model"].get("expectile", 0)
            row["weight_temp"] = config["model"].get("weight_temp", 0)
        row["random_seed"] = config.get("random_seed", 42)

        # History
        history = run.get("history", {})
        success_rates = history.get("eval_success_rates", [])
        if success_rates:
            row["latest_eval_success_rate"] = success_rates[-1]
            row["max_eval_success_rate"] = max(success_rates)
            top_3_rates = sorted(success_rates, reverse=True)[:3]
            row["top3_avg_eval_success_rate"] = sum(top_3_rates) / len(top_3_rates)
            for i, rate in enumerate(top_3_rates):
                row[f"top_eval_success_{i + 1}"] = rate
            row["history_eval_success_rates"] = success_rates
            row["history_epochs"] = history.get("epochs", [])
        rows.append(row)

    df = pd.DataFrame(rows)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at", ascending=False)
    return df


def save_summary_csv(df, output_path="iql_runs_summary.csv"):
    """Save a summary CSV file with run information."""
    columns = [
        "run_name", "user", "dataset", "algorithm", "reward_model", "use_ground_truth",
        "scale_rewards", "n_epochs", "actor_lr", "critic_lr", "expectile", "weight_temp",
        "random_seed", "top3_avg_eval_success_rate", "top_eval_success_1", "top_eval_success_2",
        "top_eval_success_3", "latest_eval_success_rate", "max_eval_success_rate", "created_at", "url"
    ]
    existing_columns = [col for col in columns if col in df.columns]
    df[existing_columns].to_csv(output_path, index=False)
    print(f"Saved summary CSV to {output_path}")

    # Top 3 only
    top3_columns = [
        "run_name", "user", "dataset", "algorithm", "expectile", "random_seed",
        "top3_avg_eval_success_rate", "top_eval_success_1", "top_eval_success_2", "top_eval_success_3", "url"
    ]
    existing_top3_columns = [col for col in top3_columns if col in df.columns]
    top3_output_path = output_path.replace(".csv", "_top3.csv")
    df[existing_top3_columns].to_csv(top3_output_path, index=False)
    print(f"Saved top 3 summary CSV to {top3_output_path}")
    return top3_output_path


def plot_algorithm_comparisons(df, output_dir="algorithm_plots"):
    """Create plots comparing algorithm performance across datasets."""
    os.makedirs(output_dir, exist_ok=True)
    if "algorithm" not in df.columns and "run_name" in df.columns:
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

    df_filtered = df.dropna(subset=["top3_avg_eval_success_rate"])
    if len(df_filtered) == 0:
        print("No runs with valid success rate metrics found. Cannot generate plots.")
        return output_dir

    print(f"Using {len(df_filtered)} runs with valid metrics out of {len(df)} total runs")
    metrics_to_plot = [
        {"column": "top_eval_success_1", "title": "Top 1 Checkpoint Success Rate"},
        {"column": "top_eval_success_2", "title": "Top 2 Checkpoint Success Rate"},
        {"column": "top_eval_success_3", "title": "Top 3 Checkpoint Success Rate"},
        {"column": "top3_avg_eval_success_rate", "title": "Average of Top 3 Checkpoints Success Rate"},
    ]
    datasets = df_filtered["dataset"].unique()
    print(f"Found {len(datasets)} datasets to plot")
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    for metric in metrics_to_plot:
        metric_column = metric["column"]
        metric_title = metric["title"]
        if metric_column not in df_filtered.columns:
            print(f"Metric {metric_column} not found in dataframe. Skipping.")
            continue
        grouped = df_filtered.groupby(["dataset", "algorithm"])
        stats = (
            grouped[metric_column]
            .agg([
                ("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max),
                ("count", "count"), ("median", np.median)
            ])
            .reset_index()
        )
        stats["ci_95"] = 1.96 * stats["std"] / np.sqrt(stats["count"].clip(1))
        for col in ["mean", "std", "min", "max", "median", "ci_95"]:
            stats[col] = stats[col].round(3)

        for dataset in datasets:
            dataset_stats = stats[stats["dataset"] == dataset]
            if len(dataset_stats) == 0:
                continue
            dataset_stats = dataset_stats.sort_values("mean", ascending=False)
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(
                x="algorithm", y="mean", data=dataset_stats, palette="viridis", alpha=0.8
            )
            for i, row in enumerate(dataset_stats.itertuples()):
                plt.errorbar(i, row.mean, yerr=row.std, fmt="none", ecolor="black", capsize=5)
                plt.text(
                    i, row.mean + 0.02, f"{row.mean:.3f}±{row.std:.3f}\nn={row.count}",
                    ha="center", va="bottom", fontsize=10,
                )
            plt.title(f"Algorithm Performance on {dataset} Dataset\n({metric_title})")
            plt.ylabel("Success Rate")
            plt.xlabel("Algorithm")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            max_value = dataset_stats["mean"].max() + dataset_stats["std"].max()
            plt.ylim(0, min(1.0, max_value * 1.2))
            safe_dataset = str(dataset).replace("/", "_")
            metric_short_name = metric_column.replace(
                "top3_avg_eval_success_rate", "top3_avg"
            ).replace("top_eval_success_", "top")
            plt.tight_layout()
            output_path = os.path.join(
                output_dir, f"{safe_dataset}_{metric_short_name}_comparison.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved {metric_short_name} plot for {dataset} to {output_path}")
            plt.close()

    create_algorithm_summary_csv(df_filtered, output_dir)
    return output_dir


def create_algorithm_summary_csv(df, output_dir="algorithm_plots"):
    """Create a summary CSV with algorithm performance by dataset."""
    metrics = [
        "top_eval_success_1", "top_eval_success_2", "top_eval_success_3", "top3_avg_eval_success_rate"
    ]
    summary_rows = []
    grouped = df.groupby(["dataset", "algorithm"])
    datasets = df["dataset"].unique()
    algorithms = df["algorithm"].unique()
    for dataset in datasets:
        row = {"Dataset": dataset}
        for alg in algorithms:
            group_data = df[(df["dataset"] == dataset) & (df["algorithm"] == alg)]
            if len(group_data) == 0:
                for metric in metrics:
                    if metric in df.columns:
                        row[f"{alg}_{metric}"] = "N/A"
                continue
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
    summary_df = pd.DataFrame(summary_rows)
    output_path = os.path.join(output_dir, "algorithm_comparison_detailed.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"Saved detailed algorithm comparison CSV to {output_path}")

    # Simplified version
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
    simplified_df = pd.DataFrame(simplified_rows)
    simplified_path = os.path.join(output_dir, "algorithm_comparison_summary.csv")
    simplified_df.to_csv(simplified_path, index=False)
    print(f"Saved simplified algorithm comparison summary CSV to {simplified_path}")
    return output_path


def plot_reward_model_comparisons(df, output_dir="reward_model_plots"):
    """Create plots comparing algorithm performance across datasets and reward models."""
    os.makedirs(output_dir, exist_ok=True)
    df_filtered = df.dropna(subset=["top3_avg_eval_success_rate"])

    # Extract reward model name and total queries
    if "reward_model" in df_filtered.columns:
        def extract_reward_model_name(path):
            if not isinstance(path, str) or path == "none":
                return "none"
            path_obj = Path(path)
            parent_dir = path_obj.parent.parent.name if "aug" in path_obj.parent.parent.name else path_obj.parent.name
            method = None
            aug = False
            if "active_" in parent_dir:
                after_active = parent_dir.split("active_", 1)[1]
                method = after_active.split("_")[0]
                if "augTrue" in parent_dir or parent_dir.endswith("_aug"):
                    aug = True
            name_parts = []
            if method:
                name_parts.append(method)
            if aug:
                name_parts.append("aug")
            return "_".join(name_parts) if name_parts else parent_dir

        def extract_total_queries(path):
            if not isinstance(path, str) or path == "none":
                return None
            try:
                path_obj = Path(path)
                checkpoint_name = path_obj.name
                match = re.search(r"iter[_\-]?(\d+)", checkpoint_name)
                if match:
                    return int(match.group(1))
            except Exception:
                pass
            match = re.search(r"max(\d+)", path)
            if match:
                return int(match.group(1))
            return None

        df_filtered["reward_model_name"] = df_filtered["reward_model"].apply(extract_reward_model_name)
        df_filtered["total_queries"] = df_filtered["reward_model"].apply(extract_total_queries)
        print(f"Extracted total_queries from {df_filtered['total_queries'].notna().sum()} reward models")
    else:
        df_filtered["reward_model_name"] = "unknown"
        df_filtered["total_queries"] = None

    if len(df_filtered) == 0:
        print("No runs with valid success rate metrics found. Cannot generate plots.")
        return output_dir

    print(f"Using {len(df_filtered)} runs with valid metrics for reward model comparison plots")
    metrics_to_plot = [
        {"column": "top3_avg_eval_success_rate", "title": "Average of Top 3 Checkpoints Success Rate"}
    ]
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})
    datasets = df_filtered["dataset"].unique()
    algorithms = df_filtered["algorithm"].unique()
    print(f"Found {len(datasets)} datasets to plot: {', '.join(datasets)}")
    print(f"Found {len(algorithms)} algorithms to plot: {', '.join(algorithms)}")

    for dataset in datasets:
        for algorithm in algorithms:
            dataset_algo_df = df_filtered[
                (df_filtered["dataset"] == dataset) & (df_filtered["algorithm"] == algorithm)
            ]
            if algorithm == "iql":
                dataset_algo_df.loc[dataset_algo_df["total_queries"].isna(), "total_queries"] = 0
                dataset_algo_df.loc[dataset_algo_df["total_queries"] == 0, "reward_model_name"] = "zero_rewards_iql"
            dataset_bc_df = df_filtered[
                (df_filtered["dataset"] == dataset) & (df_filtered["algorithm"] == "bc")
            ]
            if len(dataset_algo_df) == 0:
                continue
            if algorithm == "iql":
                query_counts = [q for q in dataset_algo_df["total_queries"].unique() if q != 0]
            else:
                query_counts = dataset_algo_df["total_queries"].unique()
            if len(query_counts) == 0:
                query_counts = [None]
            print(f"Found {len(query_counts)} unique query counts for {algorithm} on {dataset}")

            for total_queries in query_counts:
                if total_queries is not None:
                    if algorithm == "iql":
                        current_df = dataset_algo_df[
                            (dataset_algo_df["total_queries"] == total_queries) |
                            (dataset_algo_df["total_queries"] == 0)
                        ]
                    else:
                        current_df = dataset_algo_df[dataset_algo_df["total_queries"] == total_queries]
                    query_label = f"Queries: {total_queries}"
                else:
                    current_df = dataset_algo_df
                    query_label = "Unknown query count"
                reward_models = current_df["reward_model_name"].unique()
                if len(reward_models) < 1:
                    print(f"Skipping {algorithm} on {dataset} with {query_label} - no reward model: {reward_models}")
                    continue
                print(f"Creating plots for {algorithm} on {dataset} with {query_label} ({len(reward_models)} reward models)")
                for metric in metrics_to_plot:
                    metric_column = metric["column"]
                    metric_title = metric["title"]
                    if metric_column not in current_df.columns:
                        continue
                    grouped = current_df.groupby(["reward_model_name"])
                    stats = (
                        grouped[metric_column]
                        .agg([
                            ("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max),
                            ("count", "count"), ("median", np.median)
                        ])
                        .reset_index()
                    )
                    stats["ci_95"] = 1.96 * stats["std"] / np.sqrt(stats["count"].clip(1))
                    for col in ["mean", "std", "min", "max", "median", "ci_95"]:
                        stats[col] = stats[col].round(3)
                    stats = stats.sort_values("mean", ascending=False)
                    fig_width = max(12, (len(reward_models) + 1) * 1.5)
                    fig_height = 8
                    plt.figure(figsize=(fig_width, fig_height))
                    ax = sns.barplot(
                        x="reward_model_name", y="mean", data=stats, palette="viridis", alpha=0.8
                    )
                    for i, row in enumerate(stats.itertuples()):
                        plt.errorbar(i, row.mean, yerr=row.std, fmt="none", ecolor="black", capsize=5)
                        plt.text(
                            i, row.mean + 0.02, f"{row.mean:.3f}±{row.std:.3f}\nn={row.count}",
                            ha="center", va="bottom", fontsize=12,
                        )
                    if len(dataset_bc_df) > 0 and metric_column in dataset_bc_df.columns:
                        bc_values = dataset_bc_df[metric_column].dropna()
                        if len(bc_values) > 0:
                            bc_mean = bc_values.mean()
                            bc_std = bc_values.std()
                            bc_count = len(bc_values)
                            plt.axhline(
                                y=bc_mean, color="r", linestyle="--", alpha=0.7,
                                label=f"BC: {bc_mean:.3f}±{bc_std:.3f} (n={bc_count})",
                            )
                            plt.fill_between(
                                [-0.5, len(stats) - 0.5], bc_mean - bc_std, bc_mean + bc_std,
                                color="r", alpha=0.1,
                            )
                            plt.legend(loc="upper right", fontsize=12)
                    plt.title(
                        f"Reward Model Comparison: {algorithm} on {dataset}\nQueries: {int(total_queries) if total_queries is not None else 'Unknown'}",
                        fontsize=12,
                    )
                    plt.ylabel("Success Rate")
                    plt.xlabel("Reward Model")
                    plt.xticks(rotation=15, ha="right")
                    plt.subplots_adjust(bottom=0.3)
                    plt.grid(axis="y", linestyle="--", alpha=0.7)
                    max_value = stats["mean"].max() + stats["std"].max()
                    plt.ylim(0, min(1.0, max_value * 1.2))
                    safe_dataset = str(dataset).replace("/", "_")
                    safe_algorithm = str(algorithm).replace("/", "_")
                    plt.tight_layout(pad=2.0, rect=[0, 0.15, 1, 0.95])
                    output_path = os.path.join(
                        output_dir,
                        f"{safe_dataset}_{safe_algorithm}_queries{int(total_queries) if total_queries is not None else 'Unknown'}_reward_model_comparison.png",
                    )
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")
                    print(f"Saved reward model comparison plot to {output_path}")
                    plt.close()
    create_reward_model_summary_csv(df_filtered, output_dir)
    return output_dir


def create_reward_model_summary_csv(df, output_dir="reward_model_plots"):
    """Create a summary CSV with reward model performance by dataset and algorithm."""
    summary_rows = []
    if "reward_model_name" not in df.columns and "reward_model" in df.columns:
        def extract_reward_model_name(path):
            if not isinstance(path, str) or path == "none":
                return "none"
            path_obj = Path(path)
            parent_dir = path_obj.parent.name
            if any(param in parent_dir for param in [
                "n", "k", "seed", "dtw", "model", "seg", "hidden", "epochs"
            ]):
                return parent_dir
            return path_obj.name
        df["reward_model_name"] = df["reward_model"].apply(extract_reward_model_name)

    if all(col in df.columns for col in ["dataset", "algorithm", "reward_model_name"]):
        grouped = df.groupby(["dataset", "algorithm", "reward_model_name"])
        for (dataset, algorithm, reward_model), group_data in grouped:
            row = {
                "Dataset": dataset,
                "Algorithm": algorithm,
                "Reward Model": reward_model,
            }
            if "top3_avg_eval_success_rate" in df.columns:
                values = group_data["top3_avg_eval_success_rate"].dropna()
                if len(values) > 0:
                    row["Mean"] = values.mean()
                    row["Std"] = values.std()
                    row["Min"] = values.min()
                    row["Max"] = values.max()
                    row["Count"] = len(values)
                    row["Success Rate"] = f"{row['Mean']:.3f}±{row['Std']:.3f} (n={row['Count']})"
                else:
                    row["Success Rate"] = "N/A"
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if "Mean" in summary_df.columns:
            summary_df = summary_df.sort_values(
                ["Dataset", "Algorithm", "Mean"], ascending=[True, True, False]
            )
        output_path = os.path.join(output_dir, "reward_model_comparison.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"Saved reward model comparison CSV to {output_path}")
        if "Success Rate" in summary_df.columns:
            try:
                pivot_df = summary_df.pivot(
                    index=["Dataset", "Algorithm"],
                    columns="Reward Model",
                    values="Success Rate",
                )
                pivot_path = os.path.join(output_dir, "reward_model_comparison_pivot.csv")
                pivot_df.to_csv(pivot_path)
                print(f"Saved pivoted reward model comparison CSV to {pivot_path}")
            except Exception as e:
                print(f"Could not create pivoted CSV: {e}")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch and analyze wandb runs for IQL policy"
    )
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
    os.makedirs(args.output_dir, exist_ok=True)
    filters = {}
    if args.filter:
        try:
            filters = json.loads(args.filter)
        except json.JSONDecodeError:
            print(f"Error: Could not parse filter JSON: {args.filter}")
            return
    if args.user:
        filters["user"] = args.user
        print(f"Filtering for runs by user: {args.user}")

    run_data = fetch_wandb_runs(
        project=args.project,
        entity=args.entity,
        filters=filters,
        max_runs=args.max_runs,
    )
    if not run_data:
        print("No runs found matching the criteria.")
        return

    df = create_run_dataframe(run_data)
    print("\nSummary of runs with valid metrics:")
    valid_metrics_count = len(df[df["top3_avg_eval_success_rate"].notna()])
    total_runs = len(df)
    print(
        f"Total runs: {total_runs}, Runs with valid metrics: {valid_metrics_count} ({valid_metrics_count / total_runs:.1%})"
    )

    if "algorithm" in df.columns and "dataset" in df.columns:
        grouped = df.groupby(["algorithm", "dataset"])
        print("\nMetrics by algorithm and dataset:")
        for (alg, dataset), group in grouped:
            valid_count = len(group[group["top3_avg_eval_success_rate"].notna()])
            total_count = len(group)
            if valid_count > 0:
                avg_success = group["top3_avg_eval_success_rate"].mean()
                print(
                    f"{alg} on {dataset}: {valid_count}/{total_count} runs with valid metrics, avg: {avg_success:.4f}"
                )
            else:
                print(
                    f"{alg} on {dataset}: {valid_count}/{total_count} runs with valid metrics"
                )

    csv_path = os.path.join(args.output_dir, "runs_summary.csv")
    top3_csv_path = save_summary_csv(df, csv_path)

    if args.plot:
        print("\nGenerating algorithm comparison plots...")
        plot_output_dir = os.path.join(args.output_dir, args.plot_dir)
        plot_algorithm_comparisons(df, plot_output_dir)
        print(f"Plots saved to {plot_output_dir}")

        print("\nGenerating reward model comparison plots...")
        reward_model_plot_dir = os.path.join(args.output_dir, "reward_model_plots")
        plot_reward_model_comparisons(df, reward_model_plot_dir)
        print(f"Reward model comparison plots saved to {reward_model_plot_dir}")

    print(f"\nAnalysis complete. Data saved to {csv_path}")


if __name__ == "__main__":
    main()
