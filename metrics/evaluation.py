
import json
import numpy as np
import pandas as pd
import time
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import Settings, Document
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import re
from collections import defaultdict
from scipy import stats
import scipy.stats as stats

# Load env + embeddings
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
Settings.embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device}
)
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# ─────────────────────────────────────────────
# Load simulation data
# ─────────────────────────────────────────────
def load_simulation_data(directory: str) -> dict:
    sim_data = {}
    for cand_name in os.listdir(directory):
        cand_dir = os.path.join(directory, cand_name)
        if not os.path.isdir(cand_dir):
            continue
        sim_data[cand_name] = {}
        for subdir in os.listdir(cand_dir):
            path = os.path.join(cand_dir, subdir, "simulation_data.json")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sim_data[cand_name][subdir] = data
    return sim_data


def sanitize_text(text):
    text = re.sub(r'[\000-\010]|[\013-\014]|[\016-\037]', '', text)
    return text[:32767]


# ─────────────────────────────────────────────
# Candidate-level plots
# ─────────────────────────────────────────────
def plot_candidate_sentiment(candidate_name, candidate_data, candidate_dir, feedback_mode):
    sentiment_data = candidate_data.get("sentiment_data", {}).get("sentiment_tracker", {})
    if not sentiment_data:
        return

    plt.figure(figsize=(10, 6))
    for agent_name, values in sentiment_data.items():
        x = range(len(values))
        plt.plot(x, values, marker="o", label=agent_name)

    plt.xlabel("Round")
    plt.ylabel("Sentiment Score (-1 to 1)")
    plt.title(f"Sentiment Evolution: {candidate_name} – {feedback_mode}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(candidate_dir, f"sentiment_per_round_{feedback_mode}_{candidate_name}.png"))
    plt.close()


def process_candidate(candidate_index, candidate_name, sim_data, experiment_directory):
    try:
        candidate_runs = sim_data[candidate_name]
        for run_key, candidate_data in candidate_runs.items():
            feedback_mode = extract_feedback_mode_from_run_key(run_key)
            candidate_dir = os.path.join(experiment_directory, candidate_name, run_key)
            os.makedirs(candidate_dir, exist_ok=True)
            plot_candidate_sentiment(candidate_name, candidate_data, candidate_dir, feedback_mode)
        plt.close('all')
    except Exception as e:
        print(f"Error processing candidate {candidate_name}: {str(e)}")


def extract_feedback_mode_from_run_key(run_key):
    """Extract feedback mode from run key (e.g., 'sentiment_temp0.0_seed42' -> 'sentiment')"""
    return run_key.split('_')[0]


# ─────────────────────────────────────────────
# Sentiment-based metrics calculations
# ─────────────────────────────────────────────
def calculate_sentiment_variance(data):
    """Calculate polarization as sentiment variance across agents"""
    sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
    if not sentiment_data or len(sentiment_data) < 2:
        return 0.0

    # Get final sentiment scores for all agents
    final_sentiments = []
    for agent_name, sentiment_history in sentiment_data.items():
        if sentiment_history:
            final_sentiments.append(sentiment_history[-1])

    return float(np.var(final_sentiments)) if len(final_sentiments) > 1 else 0.0


def calculate_agent_synchronization(data):
    """Calculate consensus as pairwise cosine similarity between agent sentiment trajectories"""
    sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
    if not sentiment_data or len(sentiment_data) < 2:
        return 0.0

    agent_names = list(sentiment_data.keys())
    similarities = []

    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            agent_i_scores = sentiment_data[agent_names[i]]
            agent_j_scores = sentiment_data[agent_names[j]]

            if len(agent_i_scores) > 1 and len(agent_j_scores) > 1:
                # Ensure same length
                min_len = min(len(agent_i_scores), len(agent_j_scores))
                scores_i = agent_i_scores[:min_len]
                scores_j = agent_j_scores[:min_len]

                # Calculate cosine similarity
                sim = cosine_similarity([scores_i], [scores_j])[0, 0]
                if not np.isnan(sim):
                    similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def calculate_sentiment_stability(data):
    """Calculate average standard deviation of sentiment within each agent over time"""
    sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
    if not sentiment_data:
        return 0.0

    stabilities = []
    for agent_name, sentiment_history in sentiment_data.items():
        if len(sentiment_history) > 1:
            stabilities.append(np.std(sentiment_history))

    return float(np.mean(stabilities)) if stabilities else 0.0


def calculate_repetition_index(data):
    """Calculate semantic similarity between consecutive messages"""
    agent_data = data.get("agent_data", [])
    repetition_scores = []

    for agent in agent_data:
        messages = [msg["content"] for msg in agent["messages"] if msg.get("content")]
        if len(messages) > 1:
            embeddings = embed_model.encode(messages)
            similarities = []
            for i in range(1, len(embeddings)):
                sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0, 0]
                similarities.append(sim)
            repetition_scores.append(np.mean(similarities))

    return float(np.mean(repetition_scores)) if repetition_scores else 0.0


def calculate_consensus_quality(data):
    """Calculate consensus quality as inverse of sentiment variance"""
    variance = calculate_sentiment_variance(data)
    return float(1 / (1 + variance))


def calculate_communication_efficiency(data):
    """Calculate efficiency as inverse of rounds needed"""
    rounds = data.get("rounds")
    if rounds is None:
        # Estimate rounds from sentiment data
        sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
        if sentiment_data:
            rounds = max(len(values) for values in sentiment_data.values()) - 1

    if rounds is None or rounds <= 0:
        return 0.0

    return float(1 / rounds)


def calculate_agent_sentiment_stats(data):
    """Calculate agent-level sentiment statistics"""
    sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
    if not sentiment_data:
        return {}
    
    agent_stats = {}
    for agent_name, sentiment_history in sentiment_data.items():
        if sentiment_history:
            agent_stats[agent_name] = {
                'mean_sentiment': float(np.mean(sentiment_history)),
                'final_sentiment': float(sentiment_history[-1]),
                'sentiment_range': float(np.max(sentiment_history) - np.min(sentiment_history)),
                'sentiment_std': float(np.std(sentiment_history)) if len(sentiment_history) > 1 else 0.0
            }
    
    return agent_stats


# ─────────────────────────────────────────────
# Statistical analysis
# ─────────────────────────────────────────────
def analyze_statistical_significance(df, output_dir, tag=""):
    """Perform statistical tests between feedback modes"""
    if df.empty or "feedback_mode" not in df.columns:
        return

    metrics = ["rounds", "sentiment_variance", "agent_synchronization",
               "sentiment_stability", "repetition_index", "consensus_quality"]

    results = []
    modes = df["feedback_mode"].unique()

    for metric in metrics:
        if metric not in df.columns:
            continue

        df[metric] = pd.to_numeric(df[metric], errors='coerce')

        for temp in df["temperature"].unique():
            for model in df["model_name"].unique():
                temp_model_data = df[(df["temperature"] == temp) & (df["model_name"] == model)]

                for i, mode1 in enumerate(modes):
                    for mode2 in modes[i + 1:]:
                        data1 = temp_model_data[temp_model_data["feedback_mode"] == mode1][metric].dropna()
                        data2 = temp_model_data[temp_model_data["feedback_mode"] == mode2][metric].dropna()

                        if len(data1) > 1 and len(data2) > 1:
                            try:
                                t_stat, p_val = stats.ttest_ind(data1, data2)
                                pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var()) /
                                                     (len(data1) + len(data2) - 2))
                                effect_size = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

                                results.append({
                                    "metric": metric,
                                    "temperature": float(temp),
                                    "model_name": model,
                                    "mode1": mode1,
                                    "mode2": mode2,
                                    "t_statistic": float(t_stat),
                                    "p_value": float(p_val),
                                    "effect_size": float(effect_size),
                                    "significant": p_val < 0.05,
                                    "mean1": float(data1.mean()),
                                    "mean2": float(data2.mean()),
                                    "n1": len(data1),
                                    "n2": len(data2)
                                })
                            except Exception as e:
                                print(
                                    f"Warning: Could not compute statistics for {metric} between {mode1} and {mode2}: {e}")
                                continue

    if results:
        stats_df = pd.DataFrame(results)
        stats_csv = os.path.join(output_dir, f"statistical_comparisons_{tag}.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"Saved statistical analysis → {stats_csv}")
        return stats_df

    return pd.DataFrame()


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean and confidence interval for a dataset (for continuous metrics only)"""
    if len(data) == 0:
        return {
            'mean': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n': 0
        }

    data = np.array(data)
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return {
            'mean': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n': 0
        }

    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if n == 1:
        return {
            'mean': float(mean),
            'ci_lower': float(mean),
            'ci_upper': float(mean),
            'std': 0.0,
            'n': n
        }

    # Calculate confidence interval using t-distribution
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin_of_error = t_critical * (std / np.sqrt(n))

    return {
        'mean': float(mean),
        'ci_lower': float(mean - margin_of_error),
        'ci_upper': float(mean + margin_of_error),
        'std': float(std),
        'n': n
    }


def calculate_discrete_stats(data):
    """Calculate statistics for discrete metrics like rounds"""
    if len(data) == 0:
        return {
            'mode': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n': 0
        }

    data = np.array(data)
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return {
            'mode': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n': 0
        }

    from scipy import stats as scipy_stats
    mode_result = scipy_stats.mode(data, keepdims=True)
    
    return {
        'mode': float(mode_result.mode[0]) if len(mode_result.mode) > 0 else float(data[0]),
        'median': float(np.median(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'n': len(data)
    }


def aggregate_metrics_with_ci(sim_data: dict, output_dir: str):
    """Updated aggregate_metrics function with proper discrete/continuous metric handling"""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all raw data points
    rows = []
    agent_sentiment_rows = []
    
    for cand, runs in sim_data.items():
        for run_key, data in runs.items():
            # Extract feedback mode, temperature, and seed from run_key
            feedback_mode = extract_feedback_mode_from_run_key(run_key)
            parts = run_key.split('_')
            temp_str = [p for p in parts if p.startswith('temp')]
            seed_str = [p for p in parts if p.startswith('seed')]

            temperature = float(temp_str[0].replace('temp', '')) if temp_str else np.nan
            seed = int(seed_str[0].replace('seed', '')) if seed_str else np.nan

            # Calculate rounds
            rounds = data.get("rounds")
            if rounds is None:
                sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
                if sentiment_data:
                    rounds = max(len(values) for values in sentiment_data.values()) - 1
            if rounds is not None:
                rounds = int(rounds)

            # Extract model name
            exp_cfg = data.get("experiment_config", {})
            model_name = exp_cfg.get("model_name", "unknown")

            # Calculate all sentiment-based metrics
            rows.append({
                "candidate": cand,
                "mode_seed": run_key,
                "feedback_mode": feedback_mode,
                "seed": seed,
                "temperature": temperature,
                "model_name": model_name,
                "rounds": rounds if rounds is not None else np.nan,
                "sentiment_variance": calculate_sentiment_variance(data),
                "agent_synchronization": calculate_agent_synchronization(data),
                "sentiment_stability": calculate_sentiment_stability(data),
                "repetition_index": calculate_repetition_index(data),
                "consensus_quality": calculate_consensus_quality(data),
                "communication_efficiency": calculate_communication_efficiency(data),
            })
            
            # Calculate agent-level sentiment stats
            agent_stats = calculate_agent_sentiment_stats(data)
            for agent_name, stats_dict in agent_stats.items():
                agent_sentiment_rows.append({
                    "candidate": cand,
                    "mode_seed": run_key,
                    "feedback_mode": feedback_mode,
                    "seed": seed,
                    "temperature": temperature,
                    "model_name": model_name,
                    "agent_name": agent_name,
                    **stats_dict
                })

    df = pd.DataFrame(rows)
    agent_df = pd.DataFrame(agent_sentiment_rows)

    # Save seed-level data
    seed_level_csv = os.path.join(output_dir, "metrics_per_seed.csv")
    df.to_csv(seed_level_csv, index=False)
    print(f"Saved seed-level metrics → {seed_level_csv}")
    
    # Save agent-level data
    agent_csv = os.path.join(output_dir, "agent_sentiment_per_seed.csv")
    agent_df.to_csv(agent_csv, index=False)
    print(f"Saved agent sentiment per seed → {agent_csv}")

    # Aggregate metrics by condition
    continuous_metrics = ["sentiment_variance", "agent_synchronization", "sentiment_stability", 
                         "repetition_index", "consensus_quality", "communication_efficiency"]
    discrete_metrics = ["rounds"]

    grouping_cols = ["candidate", "feedback_mode", "temperature", "model_name"]
    aggregated_rows = []

    for name, group in df.groupby(grouping_cols):
        row_dict = dict(zip(grouping_cols, name))

        # Handle continuous metrics with CI
        for metric in continuous_metrics:
            if metric in group.columns:
                ci_results = calculate_confidence_interval(group[metric].dropna())
                row_dict[f"{metric}_mean"] = ci_results['mean']
                row_dict[f"{metric}_ci_lower"] = ci_results['ci_lower']
                row_dict[f"{metric}_ci_upper"] = ci_results['ci_upper']
                row_dict[f"{metric}_std"] = ci_results['std']
                row_dict[f"{metric}_n"] = ci_results['n']

        # Handle discrete metrics differently
        for metric in discrete_metrics:
            if metric in group.columns:
                discrete_results = calculate_discrete_stats(group[metric].dropna())
                row_dict[f"{metric}_mode"] = discrete_results['mode']
                row_dict[f"{metric}_median"] = discrete_results['median']
                row_dict[f"{metric}_min"] = discrete_results['min']
                row_dict[f"{metric}_max"] = discrete_results['max']
                row_dict[f"{metric}_n"] = discrete_results['n']

        aggregated_rows.append(row_dict)

    # Aggregate agent sentiment data
    agent_aggregated_rows = []
    agent_grouping_cols = grouping_cols + ["agent_name"]
    
    for name, group in agent_df.groupby(agent_grouping_cols):
        row_dict = dict(zip(agent_grouping_cols, name))
        
        agent_metrics = ["mean_sentiment", "final_sentiment", "sentiment_range", "sentiment_std"]
        for metric in agent_metrics:
            if metric in group.columns:
                ci_results = calculate_confidence_interval(group[metric].dropna())
                row_dict[f"{metric}_mean"] = ci_results['mean']
                row_dict[f"{metric}_ci_lower"] = ci_results['ci_lower'] 
                row_dict[f"{metric}_ci_upper"] = ci_results['ci_upper']
                row_dict[f"{metric}_std"] = ci_results['std']
                row_dict[f"{metric}_n"] = ci_results['n']
        
        agent_aggregated_rows.append(row_dict)

    aggregated_df = pd.DataFrame(aggregated_rows)
    agent_aggregated_df = pd.DataFrame(agent_aggregated_rows)

    # Save aggregated data
    agg_csv = os.path.join(output_dir, "aggregated_metrics_with_ci.csv")
    aggregated_df.to_csv(agg_csv, index=False)
    print(f"Saved aggregated metrics with 95% CI → {agg_csv}")
    
    agent_agg_csv = os.path.join(output_dir, "agent_sentiment_aggregated.csv")
    agent_aggregated_df.to_csv(agent_agg_csv, index=False)
    print(f"Saved aggregated agent sentiment data → {agent_agg_csv}")

    return df, aggregated_df, agent_df, agent_aggregated_df


def create_plots_with_ci(candidate_agg, output_dir, tag):
    """Create plots with confidence intervals for continuous metrics only"""
    plot_dir = os.path.join(output_dir, "plots_with_ci")
    os.makedirs(plot_dir, exist_ok=True)

    # Only continuous metrics get CI plots
    continuous_metrics_to_plot = [
        ("sentiment_variance", "Sentiment Variance (Polarization)"),
        ("agent_synchronization", "Agent Synchronization (Consensus)"),
        ("sentiment_stability", "Sentiment Stability"),
        ("repetition_index", "Repetition Index"),
        ("consensus_quality", "Consensus Quality"),
        ("communication_efficiency", "Communication Efficiency"),
    ]

    for metric, ylabel in continuous_metrics_to_plot:
        mean_col = f"{metric}_mean"
        ci_lower_col = f"{metric}_ci_lower"
        ci_upper_col = f"{metric}_ci_upper"

        if mean_col not in candidate_agg.columns:
            continue

        plt.figure(figsize=(12, 8))

        for feedback_mode in candidate_agg["feedback_mode"].unique():
            for model in candidate_agg["model_name"].unique():
                subset = candidate_agg[
                    (candidate_agg["feedback_mode"] == feedback_mode) &
                    (candidate_agg["model_name"] == model)
                    ].sort_values("temperature")

                if not subset.empty and not subset[mean_col].isna().all():
                    label = f"{model}_{feedback_mode}"

                    y_err_lower = subset[mean_col] - subset[ci_lower_col]
                    y_err_upper = subset[ci_upper_col] - subset[mean_col]
                    yerr = [y_err_lower, y_err_upper]

                    plt.errorbar(subset["temperature"], subset[mean_col],
                                 yerr=yerr, marker="o", label=label,
                                 capsize=5, capthick=2)

        plt.xlabel("Temperature")
        plt.ylabel(f"{ylabel} (Mean ± 95% CI)")
        plt.title(f"{ylabel} vs Temperature — {tag}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(plot_dir, f"{metric}_vs_temp_with_ci_{tag}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_rounds_plots(candidate_agg, output_dir, tag):
    """Create specialized plots for rounds (discrete metric)"""
    plot_dir = os.path.join(output_dir, "plots_with_ci")
    os.makedirs(plot_dir, exist_ok=True)

    if "rounds_median" not in candidate_agg.columns:
        return

    plt.figure(figsize=(12, 8))

    width = 0.8
    x_positions = np.arange(len(candidate_agg["temperature"].unique()))
    
    for i, feedback_mode in enumerate(candidate_agg["feedback_mode"].unique()):
        for j, model in enumerate(candidate_agg["model_name"].unique()):
            subset = candidate_agg[
                (candidate_agg["feedback_mode"] == feedback_mode) &
                (candidate_agg["model_name"] == model)
                ].sort_values("temperature")

            if not subset.empty:
                label = f"{model}_{feedback_mode}"
                offset = (i * len(candidate_agg["model_name"].unique()) + j) * width / len(candidate_agg["feedback_mode"].unique()) / len(candidate_agg["model_name"].unique())
                
                plt.bar(x_positions + offset, subset["rounds_median"], 
                       width=width/(len(candidate_agg["feedback_mode"].unique()) * len(candidate_agg["model_name"].unique())), 
                       label=label, alpha=0.7)
                
                # Add error bars showing min/max
                yerr_lower = subset["rounds_median"] - subset["rounds_min"]
                yerr_upper = subset["rounds_max"] - subset["rounds_median"]
                plt.errorbar(x_positions + offset, subset["rounds_median"],
                           yerr=[yerr_lower, yerr_upper], fmt='none', color='black', alpha=0.5)

    plt.xlabel("Temperature")
    plt.ylabel("Number of Rounds")
    plt.title(f"Rounds Distribution (Median with Min/Max) — {tag}")
    plt.xticks(x_positions, candidate_agg["temperature"].unique())
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f"rounds_vs_temp_{tag}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# Main eval pipeline
# ─────────────────────────────────────────────
def eval_main(experiment_directory, num_processes):
    print("Loading simulation data...")
    sim_data = load_simulation_data(experiment_directory)
    candidates = list(sim_data.keys())
    print(f"Found {len(candidates)} candidates")

    # Candidate plots
    pool = mp.Pool(processes=num_processes)
    process_func = partial(process_candidate, sim_data=sim_data, experiment_directory=experiment_directory)
    pool.starmap(process_func, enumerate(candidates))
    pool.close()
    pool.join()

    # Group by feedback mode
    feedback_mode_groups = {}
    for cand, runs in sim_data.items():
        for run_key, data in runs.items():
            feedback_mode = extract_feedback_mode_from_run_key(run_key)
            feedback_mode_groups.setdefault(feedback_mode, {}).setdefault(cand, {})[run_key] = data

    # Initialize global data collectors OUTSIDE the loop
    all_raw_dfs = []
    all_agg_dfs = []
    all_agent_raw_dfs = []
    all_agent_agg_dfs = []

    # Evaluate per feedback mode
    for feedback_mode, mode_data in feedback_mode_groups.items():
        print(f"\nEvaluating feedback mode: {feedback_mode}")
        mode_dir = os.path.join(experiment_directory, feedback_mode)
        os.makedirs(mode_dir, exist_ok=True)
        aggregate_dir = os.path.join(mode_dir, "aggregate")
        os.makedirs(aggregate_dir, exist_ok=True)

        # Calculate metrics with CI for this mode
        raw_df, agg_df, agent_raw_df, agent_agg_df = aggregate_metrics_with_ci(mode_data, aggregate_dir)
        if not raw_df.empty:
            all_raw_dfs.append(raw_df)
            all_agg_dfs.append(agg_df)
            all_agent_raw_dfs.append(agent_raw_df)
            all_agent_agg_dfs.append(agent_agg_df)

    # Global analysis across all feedback modes
    if all_raw_dfs and all_agg_dfs:
        print("\n Running comprehensive cross-condition analysis with confidence intervals...")

        # Combine all data
        combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
        combined_agg_df = pd.concat(all_agg_dfs, ignore_index=True)
        combined_agent_raw_df = pd.concat(all_agent_raw_dfs, ignore_index=True)
        combined_agent_agg_df = pd.concat(all_agent_agg_dfs, ignore_index=True)

        # Save global data
        global_agg_csv = os.path.join(experiment_directory, "global_aggregated_with_ci.csv")
        combined_agg_df.to_csv(global_agg_csv, index=False)
        print(f"Saved global aggregated data with CI → {global_agg_csv}")
        
        global_agent_csv = os.path.join(experiment_directory, "global_agent_sentiment_aggregated.csv")
        combined_agent_agg_df.to_csv(global_agent_csv, index=False)
        print(f"Saved global agent sentiment data → {global_agent_csv}")

        # Create plots
        create_plots_with_ci(combined_agg_df, experiment_directory, "global")
        create_rounds_plots(combined_agg_df, experiment_directory, "global")

        # Statistical comparisons
        analyze_statistical_significance(combined_raw_df, experiment_directory, tag="global")

    print("\n Comprehensive sentiment-based evaluation with confidence intervals complete.")





# import json
# import numpy as np
# import pandas as pd
# import time
# import torch
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from llama_index.core import Settings, Document
# import matplotlib.pyplot as plt
# import os
# import multiprocessing as mp
# from functools import partial
# from sklearn.metrics.pairwise import cosine_similarity
# import seaborn as sns
# import re
# from collections import defaultdict
# from scipy import stats
# import scipy.stats as stats

# # Load env + embeddings
# load_dotenv()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# Settings.embed_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     model_kwargs={"device": device}
# )
# embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# # ─────────────────────────────────────────────
# # Load simulation data
# # ─────────────────────────────────────────────
# def load_simulation_data(directory: str) -> dict:
#     sim_data = {}
#     for cand_name in os.listdir(directory):
#         cand_dir = os.path.join(directory, cand_name)
#         if not os.path.isdir(cand_dir):
#             continue
#         sim_data[cand_name] = {}
#         for subdir in os.listdir(cand_dir):
#             path = os.path.join(cand_dir, subdir, "simulation_data.json")
#             if os.path.isfile(path):
#                 with open(path, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                 sim_data[cand_name][subdir] = data
#     return sim_data


# def sanitize_text(text):
#     text = re.sub(r'[\000-\010]|[\013-\014]|[\016-\037]', '', text)
#     return text[:32767]


# # ─────────────────────────────────────────────
# # Candidate-level plots
# # ─────────────────────────────────────────────
# def plot_candidate_sentiment(candidate_name, candidate_data, candidate_dir, feedback_mode):
#     sentiment_data = candidate_data.get("sentiment_data", {}).get("sentiment_tracker", {})
#     if not sentiment_data:
#         return

#     plt.figure(figsize=(10, 6))
#     for agent_name, values in sentiment_data.items():
#         x = range(len(values))
#         plt.plot(x, values, marker="o", label=agent_name)

#     plt.xlabel("Round")
#     plt.ylabel("Sentiment Score (-1 to 1)")
#     plt.title(f"Sentiment Evolution: {candidate_name} – {feedback_mode}")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(candidate_dir, f"sentiment_per_round_{feedback_mode}_{candidate_name}.png"))
#     plt.close()


# def process_candidate(candidate_index, candidate_name, sim_data, experiment_directory):
#     try:
#         candidate_runs = sim_data[candidate_name]
#         for run_key, candidate_data in candidate_runs.items():
#             feedback_mode = extract_feedback_mode_from_run_key(run_key)
#             candidate_dir = os.path.join(experiment_directory, candidate_name, run_key)
#             os.makedirs(candidate_dir, exist_ok=True)
#             plot_candidate_sentiment(candidate_name, candidate_data, candidate_dir, feedback_mode)
#         plt.close('all')
#     except Exception as e:
#         print(f"Error processing candidate {candidate_name}: {str(e)}")


# def extract_feedback_mode_from_run_key(run_key):
#     """Extract feedback mode from run key (e.g., 'sentiment_temp0.0_seed42' -> 'sentiment')"""
#     return run_key.split('_')[0]


# # ─────────────────────────────────────────────
# # Sentiment-based metrics calculations
# # ─────────────────────────────────────────────
# def calculate_sentiment_variance(data):
#     """Calculate polarization as sentiment variance across agents"""
#     sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
#     if not sentiment_data or len(sentiment_data) < 2:
#         return 0.0

#     # Get final sentiment scores for all agents
#     final_sentiments = []
#     for agent_name, sentiment_history in sentiment_data.items():
#         if sentiment_history:
#             final_sentiments.append(sentiment_history[-1])

#     return float(np.var(final_sentiments)) if len(final_sentiments) > 1 else 0.0


# def calculate_agent_synchronization(data):
#     """Calculate consensus as pairwise cosine similarity between agent sentiment trajectories"""
#     sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
#     if not sentiment_data or len(sentiment_data) < 2:
#         return 0.0

#     agent_names = list(sentiment_data.keys())
#     similarities = []

#     for i in range(len(agent_names)):
#         for j in range(i + 1, len(agent_names)):
#             agent_i_scores = sentiment_data[agent_names[i]]
#             agent_j_scores = sentiment_data[agent_names[j]]

#             if len(agent_i_scores) > 1 and len(agent_j_scores) > 1:
#                 # Ensure same length
#                 min_len = min(len(agent_i_scores), len(agent_j_scores))
#                 scores_i = agent_i_scores[:min_len]
#                 scores_j = agent_j_scores[:min_len]

#                 # Calculate cosine similarity
#                 sim = cosine_similarity([scores_i], [scores_j])[0, 0]
#                 if not np.isnan(sim):
#                     similarities.append(sim)

#     return float(np.mean(similarities)) if similarities else 0.0


# def calculate_sentiment_stability(data):
#     """Calculate average standard deviation of sentiment within each agent over time"""
#     sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
#     if not sentiment_data:
#         return 0.0

#     stabilities = []
#     for agent_name, sentiment_history in sentiment_data.items():
#         if len(sentiment_history) > 1:
#             stabilities.append(np.std(sentiment_history))

#     return float(np.mean(stabilities)) if stabilities else 0.0


# def calculate_repetition_index(data):
#     """Calculate semantic similarity between consecutive messages"""
#     agent_data = data.get("agent_data", [])
#     repetition_scores = []

#     for agent in agent_data:
#         messages = [msg["content"] for msg in agent["messages"] if msg.get("content")]
#         if len(messages) > 1:
#             embeddings = embed_model.encode(messages)
#             similarities = []
#             for i in range(1, len(embeddings)):
#                 sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0, 0]
#                 similarities.append(sim)
#             repetition_scores.append(np.mean(similarities))

#     return float(np.mean(repetition_scores)) if repetition_scores else 0.0


# def calculate_consensus_quality(data):
#     """Calculate consensus quality as inverse of sentiment variance"""
#     variance = calculate_sentiment_variance(data)
#     return float(1 / (1 + variance))


# def calculate_communication_efficiency(data):
#     """Calculate efficiency as inverse of rounds needed"""
#     rounds = data.get("rounds")
#     if rounds is None:
#         # Estimate rounds from sentiment data
#         sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
#         if sentiment_data:
#             rounds = max(len(values) for values in sentiment_data.values()) - 1

#     if rounds is None or rounds <= 0:
#         return 0.0

#     return float(1 / rounds)



# # ─────────────────────────────────────────────
# # Statistical analysis
# # ─────────────────────────────────────────────
# def analyze_statistical_significance(df, output_dir, tag=""):
#     """Perform statistical tests between feedback modes"""
#     from scipy import stats
#     import pandas as pd

#     if df.empty or "feedback_mode" not in df.columns:
#         return

#     metrics = ["rounds", "sentiment_variance", "agent_synchronization",
#                "sentiment_stability", "repetition_index", "consensus_quality"]

#     results = []
#     modes = df["feedback_mode"].unique()

#     for metric in metrics:
#         if metric not in df.columns:
#             continue

#         df[metric] = pd.to_numeric(df[metric], errors='coerce')

#         for temp in df["temperature"].unique():
#             for model in df["model_name"].unique():
#                 temp_model_data = df[(df["temperature"] == temp) & (df["model_name"] == model)]

#                 for i, mode1 in enumerate(modes):
#                     for mode2 in modes[i + 1:]:
#                         data1 = temp_model_data[temp_model_data["feedback_mode"] == mode1][metric].dropna()
#                         data2 = temp_model_data[temp_model_data["feedback_mode"] == mode2][metric].dropna()

#                         if len(data1) > 1 and len(data2) > 1:
#                             try:
#                                 t_stat, p_val = stats.ttest_ind(data1, data2)
#                                 pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + (len(data2) - 1) * data2.var()) /
#                                                      (len(data1) + len(data2) - 2))
#                                 effect_size = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

#                                 results.append({
#                                     "metric": metric,
#                                     "temperature": float(temp),
#                                     "model_name": model,
#                                     "mode1": mode1,
#                                     "mode2": mode2,
#                                     "t_statistic": float(t_stat),
#                                     "p_value": float(p_val),
#                                     "effect_size": float(effect_size),
#                                     "significant": p_val < 0.05,
#                                     "mean1": float(data1.mean()),
#                                     "mean2": float(data2.mean()),
#                                     "n1": len(data1),
#                                     "n2": len(data2)
#                                 })
#                             except Exception as e:
#                                 print(
#                                     f"Warning: Could not compute statistics for {metric} between {mode1} and {mode2}: {e}")
#                                 continue

#     if results:
#         stats_df = pd.DataFrame(results)
#         stats_csv = os.path.join(output_dir, f"statistical_comparisons_{tag}.csv")
#         stats_df.to_csv(stats_csv, index=False)
#         print(f"Saved statistical analysis → {stats_csv}")
#         return stats_df

#     return pd.DataFrame()


# def calculate_confidence_interval(data, confidence=0.95):
#     """
#     Calculate mean and confidence interval for a dataset.

#     Args:
#         data: array-like of numeric values
#         confidence: confidence level (default 0.95 for 95% CI)

#     Returns:
#         dict with 'mean', 'ci_lower', 'ci_upper', 'std', 'n'
#     """
#     if len(data) == 0:
#         return {
#             'mean': np.nan,
#             'ci_lower': np.nan,
#             'ci_upper': np.nan,
#             'std': np.nan,
#             'n': 0
#         }

#     data = np.array(data)
#     data = data[~np.isnan(data)]  # Remove NaN values

#     if len(data) == 0:
#         return {
#             'mean': np.nan,
#             'ci_lower': np.nan,
#             'ci_upper': np.nan,
#             'std': np.nan,
#             'n': 0
#         }

#     n = len(data)
#     mean = np.mean(data)
#     std = np.std(data, ddof=1)  # Sample standard deviation

#     if n == 1:
#         return {
#             'mean': float(mean),
#             'ci_lower': float(mean),
#             'ci_upper': float(mean),
#             'std': 0.0,
#             'n': n
#         }

#     # Calculate confidence interval using t-distribution
#     alpha = 1 - confidence
#     t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
#     margin_of_error = t_critical * (std / np.sqrt(n))

#     return {
#         'mean': float(mean),
#         'ci_lower': float(mean - margin_of_error),
#         'ci_upper': float(mean + margin_of_error),
#         'std': float(std),
#         'n': n
#     }


# def aggregate_metrics_with_ci(sim_data: dict, output_dir: str):
#     """
#     Updated aggregate_metrics function that calculates confidence intervals across seeds.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # First, collect all raw data points
#     rows = []
#     for cand, runs in sim_data.items():
#         for run_key, data in runs.items():
#             # Extract feedback mode, temperature, and seed from run_key
#             feedback_mode = extract_feedback_mode_from_run_key(run_key)
#             parts = run_key.split('_')
#             temp_str = [p for p in parts if p.startswith('temp')]
#             seed_str = [p for p in parts if p.startswith('seed')]

#             temperature = float(temp_str[0].replace('temp', '')) if temp_str else np.nan
#             seed = int(seed_str[0].replace('seed', '')) if seed_str else np.nan

#             # Calculate rounds
#             rounds = data.get("rounds")
#             if rounds is None:
#                 sentiment_data = data.get("sentiment_data", {}).get("sentiment_tracker", {})
#                 if sentiment_data:
#                     rounds = max(len(values) for values in sentiment_data.values()) - 1
#             if rounds is not None:
#                 rounds = int(rounds)

#             # Extract model name
#             exp_cfg = data.get("experiment_config", {})
#             model_name = exp_cfg.get("model_name", "unknown")

#             # Calculate all sentiment-based metrics
#             rows.append({
#                 "candidate": cand,
#                 "mode_seed": run_key,
#                 "feedback_mode": feedback_mode,
#                 "seed": seed,
#                 "temperature": temperature,
#                 "model_name": model_name,
#                 "rounds": rounds if rounds is not None else np.nan,
#                 "sentiment_variance": calculate_sentiment_variance(data),
#                 "agent_synchronization": calculate_agent_synchronization(data),
#                 "sentiment_stability": calculate_sentiment_stability(data),
#                 "repetition_index": calculate_repetition_index(data),
#                 "consensus_quality": calculate_consensus_quality(data),
#                 "communication_efficiency": calculate_communication_efficiency(data),
#             })

#     df = pd.DataFrame(rows)

#     # Save seed-level data (raw data points)
#     seed_level_csv = os.path.join(output_dir, "metrics_per_seed.csv")
#     df.to_csv(seed_level_csv, index=False)
#     print(f"Saved seed-level metrics → {seed_level_csv}")

#     # Now aggregate with confidence intervals
#     metrics_to_aggregate = [
#         "rounds", "sentiment_variance", "agent_synchronization",
#         "sentiment_stability", "repetition_index", "consensus_quality",
#         "communication_efficiency"
#     ]

#     # Group by condition (everything except seed)
#     grouping_cols = ["candidate", "feedback_mode", "temperature", "model_name"]
#     aggregated_rows = []

#     for name, group in df.groupby(grouping_cols):
#         row_dict = dict(zip(grouping_cols, name))

#         # Calculate CI for each metric
#         for metric in metrics_to_aggregate:
#             if metric in group.columns:
#                 ci_results = calculate_confidence_interval(group[metric].dropna())
#                 row_dict[f"{metric}_mean"] = ci_results['mean']
#                 row_dict[f"{metric}_ci_lower"] = ci_results['ci_lower']
#                 row_dict[f"{metric}_ci_upper"] = ci_results['ci_upper']
#                 row_dict[f"{metric}_std"] = ci_results['std']
#                 row_dict[f"{metric}_n"] = ci_results['n']

#                 # Also include median for rounds
#                 if metric == "rounds":
#                     row_dict[f"{metric}_median"] = float(group[metric].median()) if not group[
#                         metric].isna().all() else np.nan

#         aggregated_rows.append(row_dict)

#     aggregated_df = pd.DataFrame(aggregated_rows)

#     # Save aggregated data with confidence intervals
#     agg_csv = os.path.join(output_dir, "aggregated_metrics_with_ci.csv")
#     aggregated_df.to_csv(agg_csv, index=False)
#     print(f"Saved aggregated metrics with 95% CI → {agg_csv}")

#     return df, aggregated_df


# def create_plots_with_ci(candidate_agg, output_dir, tag):
#     """Create plots with confidence intervals"""
#     plot_dir = os.path.join(output_dir, "plots_with_ci")
#     os.makedirs(plot_dir, exist_ok=True)

#     metrics_to_plot = [
#         ("rounds", "Number of Rounds"),
#         ("sentiment_variance", "Sentiment Variance (Polarization)"),
#         ("agent_synchronization", "Agent Synchronization (Consensus)"),
#         ("sentiment_stability", "Sentiment Stability"),
#         ("repetition_index", "Repetition Index"),
#         ("consensus_quality", "Consensus Quality"),
#         ("communication_efficiency", "Communication Efficiency"),
#     ]

#     for metric, ylabel in metrics_to_plot:
#         mean_col = f"{metric}_mean"
#         ci_lower_col = f"{metric}_ci_lower"
#         ci_upper_col = f"{metric}_ci_upper"

#         if mean_col not in candidate_agg.columns:
#             continue

#         plt.figure(figsize=(12, 8))

#         # Plot by feedback mode
#         for feedback_mode in candidate_agg["feedback_mode"].unique():
#             for model in candidate_agg["model_name"].unique():
#                 subset = candidate_agg[
#                     (candidate_agg["feedback_mode"] == feedback_mode) &
#                     (candidate_agg["model_name"] == model)
#                     ].sort_values("temperature")

#                 if not subset.empty and not subset[mean_col].isna().all():
#                     label = f"{model}_{feedback_mode}"

#                     # Calculate error bars (CI width)
#                     y_err_lower = subset[mean_col] - subset[ci_lower_col]
#                     y_err_upper = subset[ci_upper_col] - subset[mean_col]
#                     yerr = [y_err_lower, y_err_upper]

#                     plt.errorbar(subset["temperature"], subset[mean_col],
#                                  yerr=yerr, marker="o", label=label,
#                                  capsize=5, capthick=2)

#         plt.xlabel("Temperature")
#         plt.ylabel(f"{ylabel} (Mean ± 95% CI)")
#         plt.title(f"{ylabel} vs Temperature — {tag}")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()

#         plt.savefig(os.path.join(plot_dir, f"{metric}_vs_temp_with_ci_{tag}.png"),
#                     dpi=300, bbox_inches='tight')
#         plt.close()


# # ─────────────────────────────────────────────
# # Main eval pipeline
# # ─────────────────────────────────────────────
# def eval_main(experiment_directory, num_processes):
#     print("Loading simulation data...")
#     sim_data = load_simulation_data(experiment_directory)
#     candidates = list(sim_data.keys())
#     print(f"Found {len(candidates)} candidates")

#     # Candidate plots
#     pool = mp.Pool(processes=num_processes)
#     process_func = partial(process_candidate, sim_data=sim_data, experiment_directory=experiment_directory)
#     pool.starmap(process_func, enumerate(candidates))
#     pool.close()
#     pool.join()

#     # Group by feedback mode
#     feedback_mode_groups = {}
#     for cand, runs in sim_data.items():
#         for run_key, data in runs.items():
#             feedback_mode = extract_feedback_mode_from_run_key(run_key)
#             feedback_mode_groups.setdefault(feedback_mode, {}).setdefault(cand, {})[run_key] = data

#     all_mode_dfs = []

#     # Evaluate per feedback mode
#     for feedback_mode, mode_data in feedback_mode_groups.items():
#         print(f"\nEvaluating feedback mode: {feedback_mode}")
#         mode_dir = os.path.join(experiment_directory, feedback_mode)
#         os.makedirs(mode_dir, exist_ok=True)
#         aggregate_dir = os.path.join(mode_dir, "aggregate")
#         os.makedirs(aggregate_dir, exist_ok=True)

#         all_raw_dfs = []
#         all_agg_dfs = []

#         # Calculate metrics with CI for this mode
#         raw_df, agg_df = aggregate_metrics_with_ci(mode_data, aggregate_dir)
#         if not raw_df.empty:
#             all_raw_dfs.append(raw_df)
#             all_agg_dfs.append(agg_df)
#         # # Calculate metrics for this mode
#         # df = aggregate_metrics(mode_data, aggregate_dir)
#         # if not df.empty:
#         #     all_mode_dfs.append(df)

#      # Global analysis across all feedback modes
#     if all_raw_dfs and all_agg_dfs:
#         print("\n🔍 Running comprehensive cross-condition analysis with confidence intervals...")

#         # Combine all raw data
#         combined_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
#         combined_agg_df = pd.concat(all_agg_dfs, ignore_index=True)

#         # Save global aggregated data
#         global_agg_csv = os.path.join(experiment_directory, "global_aggregated_with_ci.csv")
#         combined_agg_df.to_csv(global_agg_csv, index=False)
#         print(f"Saved global aggregated data with CI → {global_agg_csv}")

#         # Create plots with confidence intervals
#         create_plots_with_ci(combined_agg_df, experiment_directory, "global")

#         # Statistical comparisons (using raw data)
#         analyze_statistical_significance(combined_raw_df, experiment_directory, tag="global")

#     print("\n Comprehensive sentiment-based evaluation complete.")