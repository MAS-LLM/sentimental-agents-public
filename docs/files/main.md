# main.py README

## Overview

The `main.py` file orchestrates the execution of a simulation for candidate(s) in a conversational setting. The simulation involves dialogue agents, sentiment analysis, decision-making, and opinion analysis. It uses various utility functions, templates, and external data sources to simulate a conversation, generate reports, and analyze opinions.

## Functions

### `run_simulation`

- **Input:** `agents` (List of DialogueAgent objects), `specified_topic` (String), `candidate_name` (String), `max_iters` (Integer)
- **Output:** Tuple of a String (summary) and a Dictionary (output)

Simulates a dialogue involving multiple agents, runs sentiment analysis, and returns a summary and analytics. The simulation is based on specified parameters, including the candidate's name, the conversation topic, and the maximum number of iterations.

### `fetch_agent_profiles`

- **Input:** `advisors` (List of Strings), `job_title` (String)
- **Output:** String (JSON representation of agent profiles)

Fetches agent profiles using templates and returns them in JSON format. The profiles include agent names, job titles, descriptions, priorities, and criteria.

### `simulate`

- **Input:** `candidate_name` (String), `candidate_bio` (String), `job_title` (String), `job_description` (String), `tools` (Dictionary), `advisors` (List of Dictionaries)
- **Output:** Tuple of a String (summary), a NonBayesianSentimentAgent object, a List of DialogueAgent objects, and a List (history)

Simulates a conversation for a candidate, generating summaries, sentiment data, and other analytics. The simulation involves generating agent information, system messages, and running the conversation.

### `get_simulation_output`

- **Input:** `agents` (List of DialogueAgent objects), `non_bayesian_agent` (NonBayesianSentimentAgent object), `history` (List), `output` (Dictionary)
- **Output:** Dictionary

Processes the output of a simulation, including agent data, raw history, summarized output, opinion reports, and non-bayesian sentiment data.

### `main`

- **Input:** `simulation_setup_data` (String), `candidate_csv` (String), `candidate_name` (String), `candidate_bio` (String)
- **Output:** None

Main function to run the simulation for candidate(s). It reads simulation setup data, loads candidate information, and executes the simulation. The results are saved in output files for each candidate.

## Usage

1. **Initialization**: Load environment variables, configure logging, and specify the OpenAI model.

2. **Simulation Setup Data**: Prepare a JSON file containing simulation setup data, including job title, job description, and a list of technical advisors.

3. **Candidate Data**: Provide candidate information either through a CSV file (`candidate_csv`) or directly by specifying the candidate's name and bio.

4. **Simulation Execution**: Run the main function (`main`) with the required arguments.

5. **Output Files**: Results for each candidate are saved in separate output directories. The simulation data, including summaries, agent data, sentiment analysis, and opinion reports, is stored in JSON files.

6. **Note**: Adjust parameters such as maximum iterations, OpenAI model names, and other configuration options based on specific requirements.

## Note

- Ensure that the required libraries, including `langchain`, `dotenv`, and others, are installed before running the script.

- The script is designed for simulating dialogues in a conversational setting and can be extended or modified for different use cases.