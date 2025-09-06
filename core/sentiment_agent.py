class SentimentAgent:
    """
    A class to analyze sentiments from messages using a model.
    """

    def __init__(self, agents: list):
        """
        Initialize the NonBayesianSentimentAgent with a list of agents.
        :param agents: List of agents.
        :param alpha: Learning rate, determines the weight of the new evidence.
        :param tolerance: Minimum change required for updating the prior.
        """
        self.agents = agents
        self.agent_tracker = {x.name: [0] for x in agents}  # Track sentiment for each agent
        self.change_tracker = {x.name: [0] for x in agents}  # Track sentiment change for each agent

    def update(self, speaker_idx: int) -> str:
        """
        Update the sentiment estimate for a specific agent.
        :param speaker_idx: Index of the agent in the agents list.
        :return: "Break" if the updated sentiment is the same as the prior, None otherwise.
        """
        agent = self.agents[speaker_idx]  # Get the agent
        agent_message = agent.messages[-1]  # Get the last message from the agent
        prior_sentiment = self.agent_tracker[agent.name][-1]  # Get the prior sentiment for the agent
        new_sentiment = agent_message.sentiment_data['overall_sentiment']  # Get the new evidence from the agent's message
        change = abs(new_sentiment - prior_sentiment)

        # If the current sentiment is the same as the prior, return "Break"
        if new_sentiment == prior_sentiment:
            return "Break"

        # Otherwise, update the trackers
        self.agent_tracker[agent.name].append(new_sentiment)
        self.change_tracker[agent.name].append(change)