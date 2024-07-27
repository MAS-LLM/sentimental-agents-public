class NonBayesianSentimentAgent:
    """
    A class to analyze sentiments from messages using a model.
    """

    def __init__(self, agents: list, alpha: float = 0.5, tolerance: float = 0.0001):
        """
        Initialize the NonBayesianSentimentAgent with a list of agents.
        :param agents: List of agents.
        :param alpha: Learning rate, determines the weight of the new evidence.
        :param tolerance: Minimum change required for updating the prior.
        """
        self.agents = agents
        self.agent_tracker = {x.name: [0] for x in agents}  # Track sentiment for each agent
        self.change_tracker = {x.name: [0] for x in agents}  # Track sentiment change for each agent
        self.alpha = alpha  # Learning rate
        self.tolerance = tolerance  # Minimum change required for updating the prior

    def update_sentiment_estimate(self, prior: float, new_evidence: float) -> tuple:
        """
        NonBayesian updating based on new evidence.
        :param prior: Prior sentiment value.
        :param new_evidence: New sentiment value.
        :return: Tuple of updated sentiment value and change.
        """
        updated_sentiment = self.alpha * new_evidence + (1 - self.alpha) * prior  # Calculate updated sentiment

        # Clip the updated_sentiment to be within [-1, 1]
        updated_sentiment = max(-1, min(1, updated_sentiment))

        # Check if the change is too small to make a difference
        change = abs(updated_sentiment - prior)
        if change < self.tolerance:
            return prior, change

        return updated_sentiment, change

    def update(self, speaker_idx: int) -> str:
        """
        Update the sentiment estimate for a specific agent.
        :param speaker_idx: Index of the agent in the agents list.
        :return: "Break" if the updated sentiment is the same as the prior, None otherwise.
        """
        agent = self.agents[speaker_idx]  # Get the agent
        agent_message = agent.messages[-1]  # Get the last message from the agent
        prior = self.agent_tracker[agent.name][-1]  # Get the prior sentiment for the agent
        new_evidence = agent_message.sentiment_data['overall_sentiment']  # Get the new evidence from the agent's message

        # Update the sentiment estimate
        updated_sentiment, change = self.update_sentiment_estimate(prior, new_evidence)

        # If the updated sentiment is the same as the prior, return "Break"
        if updated_sentiment == prior:
            return "Break"

        # Otherwise, update the trackers
        self.agent_tracker[agent.name].append(updated_sentiment)
        self.change_tracker[agent.name].append(change)
