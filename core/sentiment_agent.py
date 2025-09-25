from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class SentimentAgent:
    def __init__(self, agents: list):
        self.agents = agents

        # Track sentiment scores over time for each agent
        self.agent_tracker = {x.name: [0] for x in agents}
        # Track changes in sentiment between turns
        self.change_tracker = {x.name: [0] for x in agents}
        # Store current messages
        self.current_messages = {}
        self.resume_context = ""

        # Calculate initial variance for metrics
        init_sentiments = [0.0 for _ in agents]
        self.init_variance = np.var(init_sentiments) if len(init_sentiments) > 1 else 0.0

        # Sentence transformer for semantic similarity
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def set_resume_context(self, resume_context: str):
        self.resume_context = resume_context

    def update(self, speaker_idx: int) -> None:
        """Update sentiment tracking for the agent who just spoke"""
        agent = self.agents[speaker_idx]
        agent_message = agent.messages[-1]

        prior_sentiment = self.agent_tracker[agent.name][-1]
        new_sentiment = agent_message.sentiment_data["overall_sentiment"]

        change = abs(new_sentiment - prior_sentiment)
        self.agent_tracker[agent.name].append(new_sentiment)
        self.change_tracker[agent.name].append(change)
        self.current_messages[agent.name] = agent_message.content

    def finalize_round(self, feedback_mode="none"):
        """Update agents with sentiment feedback based on mode"""
        if feedback_mode == "none":
            return

        # Get current sentiment for all agents
        current_sentiments = {}
        for agent in self.agents:
            if agent.messages:
                recent_sentiment = agent.messages[-1].sentiment_data["label"]
                current_sentiments[agent.name] = recent_sentiment

        # Update agents based on feedback mode
        for agent in self.agents:
            if feedback_mode == "own_sentiment" and hasattr(agent, 'update_own_sentiment_awareness'):
                own_sentiment = current_sentiments.get(agent.name, "neutral")
                agent.update_own_sentiment_awareness(own_sentiment)

            elif feedback_mode == "others_sentiment" and hasattr(agent, 'update_others_sentiment_awareness'):
                others_sentiment = {name: sentiment for name, sentiment in current_sentiments.items()
                                    if name != agent.name}
                agent.update_others_sentiment_awareness(others_sentiment)

    def check_stopping_semantic(self, similarity_threshold=0.8) -> bool:
        """
        Semantic stopping rule: Stop when last two messages of each agent are near-duplicates.
        Stop when ALL agents are repeating themselves semantically.
        """

        # Check semantic similarity
        sims_log = {}
        repeating_flags = []
        for agent in self.agents:
            if len(agent.messages) < 2:
                return False  # need at least 2 turns

            prev = agent.messages[-2].content
            curr = agent.messages[-1].content

            prev_emb = self.encoder.encode([prev])[0]
            curr_emb = self.encoder.encode([curr])[0]
            sim = cosine_similarity([prev_emb], [curr_emb])[0, 0]
            sims_log[agent.name] = round(float(sim), 3)
            repeating_flags.append(sim >= similarity_threshold)

        semantic_stop = all(repeating_flags)

        # print(f"[Stopping Check] sims={sims_log}, semantic_stop={semantic_stop}")

        return semantic_stop

    def get_sentiment_dynamics_data(self) -> dict:
        """Get sentiment tracking data and computed metrics"""
        final_sentiments = [hist[-1] for hist in self.agent_tracker.values() if hist]
        init_sentiments = [hist[0] for hist in self.agent_tracker.values() if hist]

        variance_init = np.var(init_sentiments) if init_sentiments else 0.0
        variance_final = np.var(final_sentiments) if final_sentiments else 0.0

        # Agreement-based consensus metric
        if final_sentiments:
            pairwise_dists = [
                abs(sent_i - sent_j)
                for i, sent_i in enumerate(final_sentiments)
                for j, sent_j in enumerate(final_sentiments)
                if i < j
            ]
            mean_distance = np.mean(pairwise_dists) if pairwise_dists else 0.0
            consensus_index = 1 - mean_distance / 2.0  # normalized [0,1]
        else:
            consensus_index = 1.0

        metrics = {
            "variance_across_agents": float(variance_final),
            "mean_sentiment": float(np.mean(final_sentiments)) if final_sentiments else 0.0,
            "consensus_index": float(consensus_index),
            "polarization_index": float(
                np.max(final_sentiments) - np.min(final_sentiments)) if final_sentiments else 0.0,
            "variance_init": float(variance_init),
        }

        return {
            "sentiment_tracker": self.agent_tracker,
            "change_tracker": self.change_tracker,
            "metrics": metrics
        }