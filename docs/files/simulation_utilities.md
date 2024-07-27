# Simulation Utilities Documentation

## Overview

The `simulation_utilities.py` file provides utility functions for generating agent information, topic descriptions, system messages, and initializing dialogue agents for a simulated conversation. The utilities leverage external templates, system messages, and OpenAI models to facilitate the setup and execution of dialogue simulations.

## Functions

### `generate_agent_information`

- **Input:** `agent_names` (Dictionary), `job_title` (String)
- **Output:** Tuple of Dictionaries (agent_descriptions, agent_priorities, agent_criteria)

Generates descriptions, priorities, and criteria for agents based on provided names and job titles. The function uses templates for advisor descriptions, priorities, and criteria.

### `generate_topic`

- **Input:** `candidate_name` (String), `candidate_bio` (String), `job_title` (String), `job_description` (String)
- **Output:** String

Generates the topic of the conversation based on candidate information, job title, and job description. The function uses templates for conversation topics.

### `generate_system_messages`

- **Input:** `agent_names` (Dictionary), `agent_descriptions` (Dictionary), `agent_priorities` (Dictionary), `agent_criteria` (Dictionary), `tools` (Dictionary), `conversation_description` (String)
- **Output:** Dictionary

Generates system messages for each agent in the conversation. The messages include agent descriptions, priorities, criteria, and available tools. The function uses templates for system messages.

### `specify_topic`

- **Input:** `topic` (String), `agent_names` (Dictionary)
- **Output:** String

Makes the conversation topic more specific by providing a prompt for human input. The function uses a combination of system and human messages to achieve this, and it relies on an OpenAI model for generating content.

### `initialize_agents`

- **Input:** `agent_names` (Dictionary), `agent_system_messages` (Dictionary)
- **Output:** List of DialogueAgent Objects

Initializes dialogue agents for the conversation based on provided names and system messages. The agents are equipped with an OpenAI model for generating responses, and additional parameters such as temperature and top-k results can be adjusted.

## Usage

1. **Initialization**: Load environment variables and configure logging.

2. **Agent Information Generation**: Use the `generate_agent_information` function to obtain descriptions, priorities, and criteria for agents.

3. **Topic Generation**: Utilize the `generate_topic` function to create the initial topic of the conversation.

4. **System Message Generation**: Generate system messages for each agent using the `generate_system_messages` function.

5. **Topic Specification**: Make the conversation topic more specific using the `specify_topic` function.

6. **Agent Initialization**: Initialize dialogue agents for the conversation with the `initialize_agents` function.

7. **Dialog Simulation**: Use the initialized agents and other parameters to simulate a conversation with the help of a dialogue simulator.

8. **Note**: Adjust parameters such as model names, temperature values, and other configuration options based on specific requirements.

## Note

- Ensure that the required libraries, including `langchain`, `dotenv`, and others, are installed before running the script.

- The script is designed for simulating dialogues in a conversational setting and can be extended or modified for different use cases.