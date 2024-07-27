# Dialog README

## Overview

The `dialog.py` file contains the implementation of a dialogue simulation framework, including the definition of dialogue agents, a simulator, and additional functionality for agents equipped with tools. The goal is to facilitate conversations among agents with different roles and capabilities.

## DialogueAgent Class

The `DialogueAgent` class represents an agent participating in the dialogue. Each agent has a name, a system message that initializes the conversation, and a language model for generating responses. The class provides methods for sending and receiving messages, as well as maintaining a history of the conversation.

### Methods

- **`reset(self)`**: Resets the agent's state, clearing message history.
- **`send(self) -> str`**: Applies the language model to generate a response based on the conversation history.
- **`receive(self, name: str, message: str) -> None`**: Concatenates a message from another agent into the message history.

## DialogueSimulator Class

The `DialogueSimulator` class orchestrates the simulation, managing multiple agents and their interactions. It provides methods to reset the simulation, inject an initial message, and perform steps in the simulation.

### Methods

- **`reset(self)`**: Resets all agents in the simulation.
- **`inject(self, name: str, message: str)`**: Initiates the conversation with a message from a specified agent.
- **`step(self) -> Tuple[str, str]`**: Advances the simulation by allowing the next agent to send a message, and all agents to receive it.

## DialogueAgentWithTools Class

The `DialogueAgentWithTools` class extends the `DialogueAgent` class to support agents equipped with additional tools. It includes functionality to handle tools during message generation.

### Methods

- **`send(self) -> str`**: Applies the language model and tools to generate a response. It tracks the total token count used by the agent.

## SentimentAnalyzer Class

The `SentimentAnalyzer` class is used to analyze the sentiment of messages. It is instantiated within the `dialog.py` file and provides sentiment analysis for agent messages.

## Usage

The `dialog.py` file is intended to be utilized within the broader simulation framework, as demonstrated in the `main.py` file. Agents are initialized, and a simulation loop is executed, allowing agents to engage in a conversation. The dialogue history and agent-specific information are recorded for further analysis.

## Note

Ensure that the required dependencies, such as the OpenAI language model and sentiment analyzer, are properly configured and accessible for the dialogue simulation to function effectively.