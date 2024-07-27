# NonBayesian Updating Documentation.

## Overview

The `nonbayesian.py` file contains the implementation of the `NonBayesianSentimentAgent` class, which is designed to analyze sentiments from messages using a non-Bayesian updating approach. The class tracks sentiment estimates for individual agents and monitors changes in sentiment over time.

## NonBayesianSentimentAgent Class

The `NonBayesianSentimentAgent` class is responsible for estimating and updating sentiment values for a list of agents. The updating process is non-Bayesian, incorporating new evidence with a specified learning rate (`alpha`). The class also includes methods to track sentiment changes and check for minimal changes.

### Constructor

- **`__init__(self, agents: list)`**: Initializes the `NonBayesianSentimentAgent` with a list of agents. Initializes trackers for sentiment and sentiment changes for each agent.

### Methods

- **`update_sentiment_estimate(self, prior: float, new_evidence: float, alpha: float = 0.7, tolerance: float = 0.005) -> tuple`**: Performs non-Bayesian updating based on new evidence, returning an updated sentiment value and change.
- **`update(self, speaker_idx: int) -> str`**: Updates the sentiment estimate for a specific agent based on the agent's last message. Returns "Break" if the updated sentiment is the same as the prior, indicating minimal change.

### Attributes

- **`agents`**: List of agents for sentiment analysis.
- **`agent_tracker`**: Dictionary tracking sentiment estimates for each agent.
- **`change_tracker`**: Dictionary tracking sentiment changes for each agent.

## Usage

The `NonBayesianSentimentAgent` class can be instantiated and utilized within a broader context, such as a dialogue simulation or message analysis system. Agents update their sentiment estimates based on new evidence provided in their messages, and the class allows tracking these updates and changes over time.

## Note

Ensure that the parameters for non-Bayesian updating, such as the learning rate (`alpha`) and tolerance, are appropriately set based on the desired sensitivity to sentiment changes. The `NonBayesianSentimentAgent` class provides a simple yet effective mechanism for tracking sentiment dynamics in a dynamic conversation.