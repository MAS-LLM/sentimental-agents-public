import os
import sys
import json
import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import nltk
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import torch
from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import Settings

import multiprocessing as mp
from functools import partial

from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

# Load environment variables
load_dotenv()

# Set up device and embedding model for defensibility check
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": device})
)

def bland_altman_plot(sentiment_df: pd.DataFrame, output_dir="results"):
    """
    Generate a Bland-Altman plot comparing Single LLM vs Multiagent sentiment.
    """
    os.makedirs(output_dir, exist_ok=True)

    single = sentiment_df["single_llm_sentiment"].values
    multi = sentiment_df["multiagent_mean_sentiment"].values
    avg = (single + multi) / 2
    diff = single - multi
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    plt.figure(figsize=(8, 6))
    plt.scatter(avg, diff, edgecolors='k', c='blue')
    plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean Diff = {mean_diff:.2f}")
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', label="+1.96 SD")
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--', label="-1.96 SD")
    plt.xlabel("Average of Single and Multiagent Sentiment")
    plt.ylabel("Difference (Single - Multiagent)")
    plt.title("Bland-Altman Plot: Single vs Multiagent Sentiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bland_altman_sentiment_plot.png"))
    plt.close()

    return {
        "mean_diff": mean_diff,
        "upper_limit": mean_diff + 1.96 * std_diff,
        "lower_limit": mean_diff - 1.96 * std_diff
    }

def cosine_similarity_analysis(sentiment_df: pd.DataFrame):
    """
    Compute cosine similarity between single and multiagent sentiment vectors.
    Returns similarity score.
    """
    single_vec = sentiment_df["single_llm_sentiment"].values.reshape(1, -1)
    multi_vec = sentiment_df["multiagent_mean_sentiment"].values.reshape(1, -1)
    similarity = cosine_similarity(single_vec, multi_vec)[0, 0]
    return similarity




def compare_single_llm_vs_multiagent_sentiment(sim_data, output_dir):
    """
    Compare single LLM vs multi-agent sentiment variance and intensity for each candidate.
    Loads single LLM data from fixed path, assumes sim_data is already loaded.
    """

    # Load single LLM results
    single_llm_path = "./single_llm_control/single_llm_evaluation_results.csv"
    single_df = pd.read_csv(single_llm_path)
    single_df['candidate_name'] = single_df['candidate_name'].str.strip().str.lower()

    # Extract sentiment stats from multi-agent sim_data
    comparison_rows = []

    for candidate_name, candidate_data in sim_data.items():
        name_normalized = candidate_name.strip().lower()
        sentiment_data = candidate_data.get("non_bayesian_data", {}).get("sentiment_data", {})

        if not sentiment_data or name_normalized not in single_df["candidate_name"].values:
            continue  # Skip if data is missing

        # Multiagent final sentiments per agent
        agent_final_sentiments = [scores[-1] for scores in sentiment_data.values() if scores]
        variance_multi = np.var(agent_final_sentiments)
        mean_multi = np.mean(agent_final_sentiments)

        # Single LLM sentiment
        single_row = single_df[single_df["candidate_name"] == name_normalized].iloc[0]
        sentiment_single = float(single_row["overall_sentiment"])

        comparison_rows.append({
            "candidate": candidate_name,
            "multiagent_mean_sentiment": mean_multi,
            "multiagent_variance": variance_multi,
            "single_llm_sentiment": sentiment_single
        })

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_rows)

    # Save to CSV
    out_csv = os.path.join(output_dir, "sentiment_comparison_single_vs_multiagent.csv")
    comparison_df.to_csv(out_csv, index=False)
    print(f"Saved comparison CSV to {out_csv}")

    # Plot: Sentiment Variance (Multiagent) vs Single LLM
    plt.figure(figsize=(10, 6))
    plt.scatter(comparison_df["single_llm_sentiment"], comparison_df["multiagent_variance"], s=100, c='blue', edgecolors='k')
    for _, row in comparison_df.iterrows():
        plt.text(row["single_llm_sentiment"], row["multiagent_variance"], row["candidate"], fontsize=8, ha='right')
    plt.xlabel("Single LLM Sentiment")
    plt.ylabel("Multiagent Sentiment Variance")
    plt.title("Single LLM vs Multiagent Sentiment Variance per Candidate")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "single_vs_multiagent_sentiment_variance.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved comparison plot to {plot_path}")

    return comparison_df


def analyze_emergent_behavior(sim_data, directory):
    """Evaluate and save metrics related to emergent behaviors like groupthink, polarization, and consensus."""
    behavior_results = []

    for scenario_name, scenario_data in sim_data.items():
        print(f"Analyzing emergent behavior for {scenario_name} scenario...")
        scenario_behavior = {'Scenario': scenario_name}

        sentiment_variances = []  # Polarization
        agent_syncs = []  # Consensus or groupthink

        if isinstance(scenario_data, dict):
            non_bayesian_data = scenario_data.get("non_bayesian_data", {})
            sentiment_data = non_bayesian_data.get("sentiment_data", {})

            # 1. Sentiment Variance Across Agents: Polarization measure
            sentiment_variance_across_agents = np.var([scores[-1] for scores in sentiment_data.values()])
            sentiment_variances.append(sentiment_variance_across_agents)

            # 2. Agent Synchronization: Measure of consensus or groupthink
            final_sentiments = [np.array([scores[-1]]) for scores in sentiment_data.values() if scores]
            if len(final_sentiments) > 1:
                agent_sync = np.mean([cosine_similarity(s1.reshape(1, -1), s2.reshape(1, -1))
                                      for i, s1 in enumerate(final_sentiments)
                                      for j, s2 in enumerate(final_sentiments) if i != j])
            else:
                agent_sync = 1.0
            agent_syncs.append(agent_sync)

        scenario_behavior['Sentiment_Variance'] = np.mean(sentiment_variances)
        scenario_behavior['Agent_Synchronization'] = np.mean(agent_syncs)

        behavior_results.append(scenario_behavior)

    behavior_df = pd.DataFrame(behavior_results)
    behavior_csv_path = os.path.join(directory, "emergent_behavior_metrics.csv")
    behavior_df.to_csv(behavior_csv_path, index=False)
    print(f"Emergent behavior metrics saved to {behavior_csv_path}")

    return behavior_df
def plot_emergent_behavior(behavior_df, directory):
    """Generate plots to visualize emergent behaviors like polarization and consensus."""
    # Removing the word 'Candidate' from Scenario names
    behavior_df['Scenario'] = behavior_df['Scenario'].str.replace('Candidate ', '')

    # Define a colormap that can handle a wide variety of colors
    cmap = plt.get_cmap("viridis")  # 'viridis' is continuous and can handle many distinct values

    # Normalize the colors to ensure a unique color for each scenario
    norm = plt.Normalize(0, len(behavior_df['Scenario']))
    colors = cmap(norm(range(len(behavior_df['Scenario']))))

    # Plotting Sentiment Variance (Polarization)
    plt.figure(figsize=(10, 6))
    plt.bar(behavior_df['Scenario'], behavior_df['Sentiment_Variance'], color=colors)
    plt.xlabel('Scenario')
    plt.ylabel('Sentiment Variance')
    plt.title('Sentiment Variance Across Scenarios (Polarization)')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    sentiment_variance_path = os.path.join(directory, "sentiment_variance_plot.png")
    plt.savefig(sentiment_variance_path)
    plt.close()
    print(f"Sentiment variance plot saved to {sentiment_variance_path}")

    # Plotting Agent Synchronization (Consensus or Groupthink)
    plt.figure(figsize=(10, 6))
    plt.bar(behavior_df['Scenario'], behavior_df['Agent_Synchronization'], color=colors)
    plt.xlabel('Scenario')
    plt.ylabel('Agent Synchronization')
    plt.title('Agent Synchronization Across Scenarios (Consensus or Groupthink)')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    agent_sync_path = os.path.join(directory, "agent_synchronization_plot.png")
    plt.savefig(agent_sync_path)
    plt.close()
    print(f"Agent synchronization plot saved to {agent_sync_path}")

    # Scatter plot: Sentiment Variance vs Agent Synchronization
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        behavior_df['Sentiment_Variance'],
        behavior_df['Agent_Synchronization'],
        c=range(len(behavior_df)),  # Color points based on order
        cmap='viridis',
        s=100,
        edgecolors='k'
    )
    for i, label in enumerate(behavior_df['Scenario']):
        plt.text(behavior_df['Sentiment_Variance'][i], behavior_df['Agent_Synchronization'][i],
                 label, fontsize=8, ha='right', va='bottom')

    plt.xlabel('Sentiment Variance (Polarization)', fontsize=12)
    plt.ylabel('Agent Synchronization (Consensus)', fontsize=12)
    plt.title('Consensus vs. Polarization', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    scatter_plot_path = os.path.join(directory, "consensus_vs_polarization_scatter.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Consensus vs. Polarization scatter plot saved to {scatter_plot_path}")


def quantitative_metrics_ablation(sim_data, directory):
    """Evaluate and save quantitative metrics for the ablation scenarios."""
    metrics_results = []

    for scenario_name, scenario_data in sim_data.items():
        print(f"Evaluating {scenario_name} scenario...")
        scenario_metrics = {'Scenario': scenario_name}

        convergence_speeds = []
        sentiment_stabilities = []
        sentiment_drifts = []
        max_rounds_utilization = []
        agent_syncs = []

        if isinstance(scenario_data, dict):
            non_bayesian_data = scenario_data.get("non_bayesian_data", {})
            sentiment_data = non_bayesian_data.get("sentiment_data", {})
            change_data = non_bayesian_data.get("change", {})

            # 1. Convergence Speed: Average rounds to converge
            avg_rounds = np.mean([len(scores) for scores in sentiment_data.values()])
            convergence_speeds.append(avg_rounds)

            # 2. Sentiment Stability: Variance of sentiment changes
            sentiment_variance = np.mean([np.var(changes) for changes in change_data.values()])
            sentiment_stabilities.append(sentiment_variance)

            # 3. Sentiment Drift: Total drift across rounds
            total_drift = np.sum([np.sum(np.abs(np.diff(scores))) for scores in sentiment_data.values()])
            sentiment_drifts.append(total_drift)

            # 4. Max Rounds Utilization
            max_rounds = 20
            percentage_max_rounds = (avg_rounds / max_rounds) * 100
            max_rounds_utilization.append(percentage_max_rounds)

            # 5. Agent Synchronization: End similarity of agent sentiments
            final_sentiments = [np.array([scores[-1]]) for scores in sentiment_data.values() if scores]
            if len(final_sentiments) > 1:
                agent_sync = np.mean([cosine_similarity(s1.reshape(1, -1), s2.reshape(1, -1))
                                      for i, s1 in enumerate(final_sentiments)
                                      for j, s2 in enumerate(final_sentiments) if i != j])
            else:
                agent_sync = 1.0
            agent_syncs.append(agent_sync)

        scenario_metrics['Avg_Convergence_Speed'] = np.mean(convergence_speeds)
        scenario_metrics['Sentiment_Stability'] = np.mean(sentiment_stabilities)
        scenario_metrics['Sentiment_Drift'] = np.mean(sentiment_drifts)
        scenario_metrics['Max_Rounds_Utilization'] = np.mean(max_rounds_utilization)
        scenario_metrics['Agent_Synchronization'] = np.mean(agent_syncs)

        metrics_results.append(scenario_metrics)

    metrics_df = pd.DataFrame(metrics_results)
    metrics_df.to_csv(os.path.join(directory, "ablation_quantitative_metrics.csv"), index=False)
    print(f"Quantitative metrics saved to {directory}/ablation_quantitative_metrics.csv")




def load_simulation_data(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    sorted_folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    sim_data = {}
    for folder in sorted_folders:
        with open(os.path.join(directory, folder, "simulation_data.json"), "r", encoding="utf-8") as jfile:
            data = json.load(jfile)
        sim_data[folder.split("_")[-1]] = data
    return sim_data


def generate_short_forms(roles):
    key = {}
    for role in roles:
        words = role.split()
        if len(words) == 1:
            short_form = role
        else:
            short_form = ''.join(word[0].upper() for word in words if word.lower() not in ['and', 'the', 'of'])
        key[role] = short_form
    return key


def prolificness_score(sim_data, directory):
    num_candidates = len(sim_data)
    rows, cols = math.ceil(num_candidates / 3), 3
    fig_width, fig_height = cols * 4, rows * 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    for i, (candidate_name, candidate_data) in enumerate(sim_data.items()):
        unique_arguments_count_by_role = {}
        roles = []
        for entry in candidate_data["agent_data"]:
            statements = [x["content"] for x in entry["messages"]]
            role = entry["name"]
            roles.append(role)
            unique_arguments_count_by_role[role] = len(statements)

        i_val, j_val = divmod(i, 3)
        ax = axes[i_val, j_val] if rows > 1 else axes[j_val]

        key = generate_short_forms(set(roles))
        roles = [key[x] for x in unique_arguments_count_by_role.keys()]
        counts = list(unique_arguments_count_by_role.values())
        ax.bar(roles, counts)
        ax.set_xlabel('Role')
        ax.set_ylabel('Number of Unique Arguments')
        ax.set_title(f'Prolificness Score for {candidate_name}')
        for k, count in enumerate(counts):
            ax.text(k, count, str(count), ha='center', va='bottom')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "prolificness_score_by_role_grid.png"))
    plt.close()  # Close the figure to free memory


def nuance_score(sim_data, directory):
    dfs = []
    candidate_names = []
    for candidate_name, candidate_data in sim_data.items():
        statements = []
        for entry in candidate_data["agent_data"]:
            statements.extend([x["content"] for x in entry["messages"]])

        additional_exclude_words = set(candidate_name.split(" "))
        stop_words = set(stopwords.words('english')).union(additional_exclude_words)
        data_words = [simple_preprocess(statement, deacc=True) for statement in statements]
        data_words_nostops = [[word for word in doc if word not in stop_words] for doc in data_words]

        id2word = Dictionary(data_words_nostops)
        corpus = [id2word.doc2bow(text) for text in data_words_nostops]

        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=5, random_state=100,
                             update_every=1, chunksize=100, passes=10, alpha='auto')

        top_words_per_topic = {f"Topic {i}": [word for word, _ in lda_model.show_topic(i, 10)]
                               for i in range(lda_model.num_topics)}

        df = pd.DataFrame(top_words_per_topic)
        dfs.append(df)
        candidate_names.append(candidate_name)

    with pd.ExcelWriter(os.path.join(directory, 'nuance_scores.xlsx')) as writer:
        for i, df in enumerate(dfs):
            df.to_excel(writer, sheet_name=candidate_names[i], index=False)


def similarity_score(data, output_directory, candidate_index):
    messages_by_agent = defaultdict(list)
    for entry in data["agent_data"]:
        agent_messages = [x["content"] for x in entry["messages"]]
        agent = entry["name"]
        messages_by_agent[agent].extend(agent_messages)

    def compute_cosine_similarity(messages):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(messages)
        return cosine_similarity(tfidf_matrix)

    # Calculate intra-agent similarities
    intra_agent_similarities = {agent: compute_cosine_similarity(messages)
                                for agent, messages in messages_by_agent.items() if messages}

    vectorizer = TfidfVectorizer()
    agents = list(messages_by_agent.keys())
    inter_agent_similarities = np.zeros((len(agents), len(agents)))

    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            if i != j and messages_by_agent[agent1] and messages_by_agent[agent2]:
                similarities = []
                for message1 in messages_by_agent[agent1]:
                    for message2 in messages_by_agent[agent2]:
                        combined_tfidf = vectorizer.fit_transform([message1, message2])
                        cosine_sim = cosine_similarity(combined_tfidf)
                        similarities.append(cosine_sim[0, 1])
                inter_agent_similarities[i, j] = np.mean(similarities) if similarities else 0

    # Only generate plots for the first 10 candidates
    if candidate_index < 10:
        # Plot and save inter-agent similarities
        plt.figure(figsize=(8, 6))
        cax = plt.matshow(inter_agent_similarities, cmap="coolwarm", vmin=0, vmax=1)
        plt.title("Inter-Agent Similarities")
        plt.xticks(range(len(agents)), agents, rotation=90, fontsize=6)
        plt.yticks(range(len(agents)), agents, fontsize=6)
        plt.colorbar(cax, fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(output_directory, "inter_agent_similarities.png"), dpi=300)
        plt.close()

        # Plot and save intra-agent similarities
        for agent, similarity in intra_agent_similarities.items():
            plt.figure(figsize=(8, 6))
            cax = plt.matshow(similarity, cmap="coolwarm", vmin=0, vmax=1)
            plt.title(f'Intra-agent Similarity - {agent}')
            plt.colorbar(cax)
            plt.savefig(os.path.join(output_directory, f"intra_agent_similarity_{agent}.png"), dpi=300)
            plt.close()

    return inter_agent_similarities, intra_agent_similarities


def sentiment_non_bayesian_plot(candidate_name, candidate_data, candidate_dir, candidate_index):
    """Plot and save sentiment and non-Bayesian change data for each candidate."""
    agents = candidate_data.get("agent_data", [])
    if not agents:
        print(f"No agent data available for {candidate_name}. Skipping.")
        return

    non_bayesian_data = candidate_data.get("non_bayesian_data", {})
    sentiment_data = non_bayesian_data.get("sentiment_data", {})
    change_data = non_bayesian_data.get("change", {})

    # Determine the maximum number of rounds
    max_rounds = max(len(scores) for scores in sentiment_data.values())

    # Only generate plots for the first 10 candidates
    if candidate_index < 10:
        # Plot and save sentiment scores
        plt.figure(figsize=(10, 6))

        for agent in agents:
            agent_name = agent.get("name", "Unknown")
            sentiment_scores = sentiment_data.get(agent_name, [])
            plt.plot(range(1, len(sentiment_scores) + 1), sentiment_scores, marker='o', linestyle='-', label=agent_name)

        plt.title(f'Sentiment Scores over Rounds for {candidate_name}', fontsize=18)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Sentiment Score', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, max_rounds + 1), [str(i) for i in range(1, max_rounds + 1)])
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(candidate_dir, f'sentiment_scores_{candidate_name}.png'), dpi=150)
        plt.close()

        # Plot and save non-Bayesian changes
        plt.figure(figsize=(12, 6))  # Increased figure width
        bar_width = 0.15  # Reduced bar width
        num_agents = len(agents)
        index = np.arange(1, max_rounds)  # Increased spacing between round groups

        for i, agent in enumerate(agents):
            agent_name = agent.get("name", "Unknown")
            change_scores = change_data.get(agent_name, [])[1:]  # Exclude round 0
            plt.bar(index[:len(change_scores)] + i * bar_width, change_scores, bar_width, label=agent_name)

        plt.title(f'Non-Bayesian Change over Rounds for {candidate_name}', fontsize=18)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Change Score', fontsize=14)
        plt.xticks(index + bar_width * (num_agents - 1) / 2, [str(i) for i in range(1, max_rounds)])
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(candidate_dir, f'non_bayesian_change_{candidate_name}.png'), dpi=150)
        plt.close()

    return sentiment_data, change_data
def calculate_drift(sim_data, directory):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    distance_model = SentenceEmbeddings(embedding_model)
    model = PolyFuzz(distance_model)

    drift_list = []
    for candidate_name, candidate_data in sim_data.items():
        agent_drift = {"candidate": candidate_name}
        for entry in candidate_data["agent_data"]:
            messages = [x["content"] for x in entry["messages"]]
            agent = entry["name"]
            if not messages:
                agent_drift[agent] = np.nan
            else:
                system_messages = candidate_data["initial_conditions"]["agent_system_messages"][agent]
                df = model.match(messages, [system_messages]).get_matches()
                agent_drift[agent] = df["Similarity"].mean()
        drift_list.append(agent_drift)

    drift_df = pd.DataFrame(drift_list)
    drift_df.to_csv(os.path.join(directory, "drift_df.csv"), index=False)


def load_resume(text: str) -> list:
    """Load resume data from text."""
    documents = [Document(text=text)]
    return documents

import re

def sanitize_text(text):
    # Remove or replace illegal characters
    text = re.sub(r'[\000-\010]|[\013-\014]|[\016-\037]', '', text)
    # Truncate to Excel's character limit (32,767 characters)
    return text[:32767]

def run_defensibility_check(directory: str, resume_file: str, export: bool = True) -> list:
    """Run defensibility check on folders in a directory."""
    resume_df = pd.read_csv(resume_file)
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    sorted_folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(directory, f)))

    dfs_list = []
    sheet_names = []
    for folder_name in sorted_folders:
        candidate_name = folder_name.split("_")[-1]
        resume_row = resume_df[resume_df['candidate_name'].str.strip().str.lower() == candidate_name.strip().lower()]

        if resume_row.empty:
            print(f"No resume found for candidate {candidate_name}. Skipping.")
            continue

        resume = resume_row.iloc[0]["resume"]

        with open(os.path.join(directory, folder_name, "simulation_data.json"), "r", encoding="utf-8") as jfile:
            data = json.load(jfile)

        documents = load_resume(resume)
        node_parser = MarkdownNodeParser.from_defaults()
        Settings.node_parser = node_parser
        Settings.embed_model = embed_model

        index = VectorStoreIndex.from_documents(documents, show_progress=True)

        retriever = index.as_retriever()
        def_list = []
        for agent in data["agent_data"]:
            agent_name = agent["name"]
            for message in agent["messages"]:
                message_text = sanitize_text(message["content"])
                response = retriever.retrieve(message_text)
                if len(response) == 0:
                    def_list.append({
                        "agent": agent_name,
                        "argument": message_text,
                        "source_text": "",
                        "score": 0
                    })
                else:
                    def_list.append({
                        "agent": agent_name,
                        "argument": message_text,
                        "source_text": sanitize_text(response[0].node.text),
                        "score": response[0].score
                    })
        def_df = pd.DataFrame(def_list)
        dfs_list.append(def_df)
        sheet_names.append(candidate_name)
        print(f"Processed {candidate_name}")

    if export:
        with pd.ExcelWriter(os.path.join(directory, "defensibility_scores.xlsx"), engine='openpyxl') as writer:
            for idx, df in enumerate(dfs_list):
                sheet_name = sheet_names[idx]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    return dfs_list


def process_candidate(candidate_index, candidate_name, sim_data, experiment_directory):
    try:
        candidate_data = sim_data[candidate_name]
        candidate_dir = os.path.join(experiment_directory, candidate_name)
        if not os.path.exists(candidate_dir):
            os.makedirs(candidate_dir)

        similarity_score(candidate_data, candidate_dir, candidate_index)
        sentiment_non_bayesian_plot(candidate_name, candidate_data, candidate_dir, candidate_index)

        # Clear matplotlib's figure cache
        plt.close('all')
    except Exception as e:
        print(f"Error processing candidate {candidate_name}: {str(e)}")


def evaluate_cognitive_bias(sentiment_df: pd.DataFrame, output_dir="results"):
    """
    Evaluates and plots cognitive bias between Single LLM and Multiagent sentiment data.
    Assumes the dataframe contains:
    - candidate
    - multiagent_mean_sentiment
    - multiagent_variance
    - single_llm_sentiment
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compute Bias Gap
    sentiment_df["bias_gap"] = sentiment_df["single_llm_sentiment"] - sentiment_df["multiagent_mean_sentiment"]

    # 2. Compute Extremity (absolute sentiment)
    sentiment_df["single_extremity"] = sentiment_df["single_llm_sentiment"].abs()
    sentiment_df["multi_extremity"] = sentiment_df["multiagent_mean_sentiment"].abs()

    # 3. Inconsistency Index
    sentiment_df["inconsistency_index"] = sentiment_df["bias_gap"].abs() + sentiment_df["multiagent_variance"]

    # Summary stats
    avg_bias_gap = sentiment_df["bias_gap"].mean()
    avg_variance = sentiment_df["multiagent_variance"].mean()
    avg_extremity_diff = (sentiment_df["single_extremity"] - sentiment_df["multi_extremity"]).mean()

    print("📊 Cognitive Bias Summary:")
    print(f"- Average Bias Gap: {avg_bias_gap:.4f}")
    print(f"- Average Multiagent Variance: {avg_variance:.4f}")
    print(f"- Avg Extremity Difference (Single - Multi): {avg_extremity_diff:.4f}")

    # === Plot 1: Bias Gap per Candidate ===
    plt.figure(figsize=(10, 6))
    plt.bar(sentiment_df["candidate"], sentiment_df["bias_gap"], color='skyblue', edgecolor='k')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Bias Gap (Single LLM - Multiagent Mean) per Candidate")
    plt.ylabel("Bias Gap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_gap_per_candidate.png"))
    plt.close()

    # === Plot 2: Inconsistency Index vs Candidate ===
    plt.figure(figsize=(10, 6))
    plt.bar(sentiment_df["candidate"], sentiment_df["inconsistency_index"], color='salmon', edgecolor='k')
    plt.title("Inconsistency Index (|Bias Gap| + Variance) per Candidate")
    plt.ylabel("Inconsistency Index")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inconsistency_index_per_candidate.png"))
    plt.close()

    # === Plot 3: Scatter of Bias Gap vs Variance ===
    plt.figure(figsize=(8, 6))
    plt.scatter(sentiment_df["bias_gap"].abs(), sentiment_df["multiagent_variance"], c='purple', edgecolors='k', s=100)
    for i, row in sentiment_df.iterrows():
        plt.text(abs(row["bias_gap"]), row["multiagent_variance"], row["candidate"],
                 fontsize=8, ha='right', va='bottom')
    plt.xlabel("Absolute Bias Gap |Single - Multi|")
    plt.ylabel("Multiagent Sentiment Variance")
    plt.title("Bias Gap vs Multiagent Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_gap_vs_variance_scatter.png"))
    plt.close()

    return sentiment_df[["candidate", "bias_gap", "multiagent_variance", "inconsistency_index"]]

def analyze_single_vs_multi_bias_regression(sentiment_df: pd.DataFrame, output_dir="results"):
    """
    Uses linear regression to analyze how single LLM sentiment relates to multiagent sentiment,
    and how bias gap relates to multiagent disagreement (variance).
    Saves plots and returns regression stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    # === A. Regress Multiagent Mean ~ Single Sentiment ===
    X1 = sentiment_df["single_llm_sentiment"].values.reshape(-1, 1)
    y1 = sentiment_df["multiagent_mean_sentiment"].values
    model1 = LinearRegression()
    model1.fit(X1, y1)
    slope1 = model1.coef_[0]
    intercept1 = model1.intercept_
    r_squared1 = model1.score(X1, y1)

    plt.figure(figsize=(8, 6))
    plt.scatter(X1, y1, c='blue', edgecolors='k')
    plt.plot(X1, model1.predict(X1), color='red', label=f'y={slope1:.2f}x + {intercept1:.2f} (R²={r_squared1:.2f})')
    plt.title("Multiagent Sentiment vs Single LLM Sentiment")
    plt.xlabel("Single LLM Sentiment")
    plt.ylabel("Multiagent Mean Sentiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_multi_vs_single.png"))
    plt.close()

    # === B. Regress |Bias Gap| ~ Multiagent Variance ===
    X2 = sentiment_df["multiagent_variance"].values.reshape(-1, 1)
    y2 = sentiment_df["bias_gap"].abs().values
    model2 = LinearRegression()
    model2.fit(X2, y2)
    slope2 = model2.coef_[0]
    intercept2 = model2.intercept_
    r_squared2 = model2.score(X2, y2)

    plt.figure(figsize=(8, 6))
    plt.scatter(X2, y2, c='purple', edgecolors='k')
    plt.plot(X2, model2.predict(X2), color='green', label=f'y={slope2:.2f}x + {intercept2:.2f} (R²={r_squared2:.2f})')
    plt.title("Absolute Bias Gap vs Multiagent Variance")
    plt.xlabel("Multiagent Variance")
    plt.ylabel("Absolute Bias Gap")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_biasgap_vs_variance.png"))
    plt.close()

    # Return summary
    return {
        "multi_vs_single": {"slope": slope1, "intercept": intercept1, "r_squared": r_squared1},
        "biasgap_vs_variance": {"slope": slope2, "intercept": intercept2, "r_squared": r_squared2},
    }

def main(experiment_directory, resume_file, num_processes):
    print("Loading simulation data...")
    sim_data = load_simulation_data(experiment_directory)

    print("Calculating nuance score...")
    # nuance_score(sim_data, experiment_directory)

    # candidates = list(sim_data.keys())

    # Print the names of the first 10 candidates (or fewer if there are less than 10)
    # plot_candidates = candidates[:10]
    # print("Generating plots for the following candidates:")
    # for i, candidate in enumerate(plot_candidates, 1):
    #     print(f"{i}. {candidate}")

    # print(f"\nProcessing all {len(candidates)} candidates using {num_processes} processes...")

    start_time = time.time()

    # Create a pool of worker processes
    pool = mp.Pool(processes=num_processes)

    # Create a partial function with fixed arguments
    process_func = partial(process_candidate, sim_data=sim_data, experiment_directory=experiment_directory)

    # Map the function to all candidates, passing the candidate index and name
    pool.starmap(process_func, enumerate(candidates))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    end_time = time.time()
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")

    # print("Calculating drift scores...")
    # calculate_drift(sim_data, experiment_directory)

    # print("Running defensibility check...")
    # run_defensibility_check(experiment_directory, resume_file)

    # **New addition for ablation metrics**
    # print("Evaluating quantitative metrics for ablation study...")
    # quantitative_metrics_ablation(sim_data, experiment_directory)

    # **New addition for emergent behavior analysis**
    # print("Analyzing emergent behaviors...")
    # behavior_df = analyze_emergent_behavior(sim_data, experiment_directory)
    # plot_emergent_behavior(behavior_df, experiment_directory)

    print("Comparing multi-agent and single LLM sentiment results...")
    compare_single_llm_vs_multiagent_sentiment(sim_data, experiment_directory)

    # Load the sentiment comparison data saved by the comparison function
    sentiment_csv_path = os.path.join(experiment_directory, "sentiment_comparison_single_vs_multiagent.csv")
    sentiment_df = pd.read_csv(sentiment_csv_path)

    bias_df = evaluate_cognitive_bias(sentiment_df, output_dir=experiment_directory)
    bias_df.to_csv(os.path.join(experiment_directory, "cognitive_bias_metrics.csv"), index=False)

    regression_stats = analyze_single_vs_multi_bias_regression(sentiment_df, output_dir=experiment_directory)
    regression_stats_df = pd.DataFrame.from_dict(regression_stats, orient='index')
    regression_stats_df.to_csv(os.path.join(experiment_directory, "regression_analysis.csv"))

    # Load the CSV and run both methods as an example
    bland_stats = bland_altman_plot(sentiment_df, output_dir=experiment_directory)
    cosine_sim = cosine_similarity_analysis(sentiment_df)
    print(f"Bland-Altman stats: {bland_stats}")
    print(f"Cosine similarity between single and multiagent sentiment: {cosine_sim:.4f}")
    # save the cosine similarity to a file
    with open(os.path.join(experiment_directory, "cosine_similarity.txt"), "w") as f:
        f.write(f"Cosine similarity: {cosine_sim:.4f}")
    # save bland-altman stats to a file
    bland_stats.to_csv(os.path.join(experiment_directory, "bland_altman_stats.csv"), index=False)


    print("Analysis complete!")



if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python script_name.py <experiment_directory> <resume_file> [num_processes]")
        sys.exit(1)

    experiment_directory = sys.argv[1]
    resume_file = sys.argv[2]

    if len(sys.argv) == 4:
        num_processes = int(sys.argv[3])
    else:
        # Use the number of CPU cores minus 1, or 1 if there's only one core
        num_processes = max(1, mp.cpu_count() - 1)

    main(experiment_directory, resume_file, num_processes)