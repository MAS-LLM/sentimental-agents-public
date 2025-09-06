from typing import List, Dict, Callable, Tuple, Any, Union
import multiprocessing as mp
import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent / "metrics"))
sys.path.append(str(pathlib.Path(__file__).parent / "single_llm_control"))
import logging
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from core.dialog import DialogueSimulator, DialogueAgent, DialogueAgentWithTools
from utilities.utilities import summarise_document
from single_llm_control.evaluate_single_llm import generate_response_from_sample
import pandas as pd
import json
from core.simulation_utilities import generate_agent_information, generate_system_messages, generate_topic, \
    specify_topic, initialize_agents
from core.sentiment_agent import SentimentAgent
from langchain.callbacks import get_openai_callback
from utilities.opinion_analyser import AdvisorReport
import argparse
import datetime
import random
from metrics.evaluation import eval_main
from tqdm import tqdm
import warnings
import logging
# import numpy as np
# ─── 1) Global Python warnings ─────────────────────────────────────────────────
warnings.filterwarnings("ignore")                              # hide UserWarning, DeprecationWarning, etc.
warnings.filterwarnings("ignore", category=DeprecationWarning)  # specifically hide DeprecationWarning

# ─── 2) Silence specific noisy loggers ──────────────────────────────────────────
for noisy in [
    "pydantic",    # Pydantic deprecation spam
    "langchain",   # LangChain INFOs
    "openai",      # OpenAI HTTP logs
    "urllib3",     # HTTP request logs
    "httpx",       # httpx logs (if used)
    "transformers",# HuggingFace transformers warnings
]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ─── 3) Transformers internal verbosity ─────────────────────────────────────────
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# ─── 4) (Optional) Quiet CUDA / PyTorch info ────────────────────────────────────
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
logging.getLogger("torch").setLevel(logging.ERROR)



class Config:
    def __init__(self):
        self.dialog_temp = 0.0
        self.specifyTopic_temp = 0.0
        self.generateContent_temp = 0.0  # Content gen for agent messages TODO: split parameters for individual agents
        self.summarize_temp = 0
        self.max_rounds = 1

    def __str__(self):
        return (f"Config(\n"
                f"    dialog_temp={self.dialog_temp},\n"
                f"    specifyTopic_temp={self.specifyTopic_temp},\n"
                f"    generateContent_temp={self.generateContent_temp},\n"
                f"    summarize_temp={self.summarize_temp},\n"
                f"    max_rounds={self.max_rounds}\n"
                f")")

    def to_dict(self):
        return {
            "dialog_temp": self.dialog_temp,
            "specifyTopic_temp": self.specifyTopic_temp,
            "generateContent_temp": self.generateContent_temp,
            "summarize_temp": self.summarize_temp,
            "max_rounds": self.max_rounds
        }


# Usage:
config = Config()

# print(config)
# print(config.to_dict())

def run_simulation(agents: List[DialogueAgent], specified_topic: str, candidate_name, config=None) -> Tuple[
    str, Dict[str, Any]]:
    """Run the simulation and return the summary and analytics."""

    seed = 42  # or pass in as argument later
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
        """Function to determine the next speaker."""
        return step % len(agents)

    if config is None:
        sentiment_agent = SentimentAgent(agents)
    else:
        sentiment_agent = SentimentAgent(agents)

    simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
    simulator.reset()
    simulator.inject("Facilitator", specified_topic)
    # Define the maximum rounds and initialize the round counter
    # max_rounds = 40  # You can adjust this value based on your needs
    round_counter = 0

    while True:
        round_counter += 1  # Increment the round counter

        # Iterate through each agent in the simulation
        for i in range(len(agents)):
            name, agent_message, speaker_idx = simulator.step()  # Get new data from the simulator

            # print('round:', round_counter, 'speaker:', name,  'speaker_idx:', speaker_idx)

            # Update the agent's sentiment and check if the stopping condition is met
            if sentiment_agent.update(speaker_idx) == "Break":
                # print(f"Agent {name} has triggered the stopping condition, ending simulation.")
                break  # Exit the for-loop if the stopping condition is met

        else:
            # If no stopping condition was met, check if the maximum rounds have been reached
            if round_counter >= config.max_rounds:
                print(f"Maximum rounds of {config.max_rounds} reached, stopping simulation.")
                break  # Exit the while-loop if the maximum rounds are reached
            continue  # Continue to the next round if no break was encountered

        break  # If an agent triggered a break, exit the while-loop

    # Post-process conversation for analytics
    history = simulator.conversation_history

    if config is None:
        summary = summarise_document(history)
    else:
        summary = summarise_document(history, config.summarize_temp)

    output = {
        "Candidate Name": candidate_name,
        "Summary": summary,
    }

    return output, sentiment_agent, history


def fetch_agent_profiles(advisors: List[str], job_title: str) -> str:
    """
    Fetches agent's profiles using templates and returns it in JSON format.
    """
    agent_profiles = []

    # Generating agent information from templates
    agent_descriptions, agent_priorities, agent_criteria = generate_agent_information(
        {advisor: None for advisor in advisors},  # Passing None for tools as it's not used here
        job_title
    )

    for advisor in advisors:
        agent_profile = {
            "Agent Name": advisor,
            "Job Title": "Technical Advisor",  # This is a simplification based on the given code
            "Description": agent_descriptions[advisor],
            "Priorities": agent_priorities[advisor],
            "Criteria": agent_criteria[advisor]
        }
        agent_profiles.append(agent_profile)

    return json.dumps(agent_profiles, indent=2)


def simulate(
        candidate_name: str,
        candidate_bio: str,
        job_title: str,
        job_description: str,
        tools: Dict,
        advisors: List[Dict],
        config=None
) -> Tuple[str, Dict[str, Any]]:
    logging.info("Starting the simulation.")

    agent_names = {advisor["title"]: tools for advisor in advisors}

    agent_descriptions, agent_priorities, agent_criteria = generate_agent_information(agent_names, job_title)

    topic = generate_topic(candidate_name, candidate_bio, job_title, job_description)
    conversation_description = f"""Here is the topic of conversation: {topic}
    The participants are: {', '.join(agent_names.keys())}"""

    agent_system_messages = generate_system_messages(agent_names, agent_descriptions, agent_priorities, agent_criteria,
                                                     tools, conversation_description)

    if config is not None:
        specified_topic = specify_topic(topic, agent_names, config.specifyTopic_temp)
    else:
        specified_topic = specify_topic(topic, agent_names)

    initial_conditions = {
        "agent_names": agent_names,
        "agent_descriptions": agent_descriptions,
        "agent_priorities": agent_priorities,
        "agent_criteria": agent_criteria,
        "topic": topic,
        "agent_system_messages": agent_system_messages,
        "specified_topic": specified_topic,
        "candidate_name": candidate_name,
        "candidate_bio": candidate_bio,
        "job_title": job_title,
        "job_description": job_description,
        "advisors": advisors,

    }
    if config is not None:
        agents = initialize_agents(agent_names, agent_system_messages, temperature=config.dialog_temp)
    else:
        agents = initialize_agents(agent_names, agent_system_messages)
    output, sentiment_agent, history = run_simulation(agents, specified_topic, candidate_name=candidate_name,
                                                         config=config)

    return output, sentiment_agent, agents, history, initial_conditions


def get_simulation_output(agents, sentiment_agent, history, output):
    # dm = DecisionMaker(agents)
    # decision_metrics = {
    #    x.name: x.decision_metrics for x in dm.agents
    # }
    # report = AdvisorReport(agents)
    agent_data = [{
        "name": agent.name,
        "messages": [x.to_dict() for x in agent.messages],
    } for agent in agents]
    out = {
        "agent_data": agent_data,
        "raw_history": history,
        "summarized_output": output,
        # "opinion_report": report.generate().to_dict(orient="records"),
        # "decision_metrics": decision_metrics,
        "sentiment_data": {
            "change": sentiment_agent.change_tracker,
            "sentiment_data": sentiment_agent.agent_tracker,
        }
    }
    return out


def main(simulation_setup_data, candidate_csv=None, candidate_name=None, candidate_bio=None, config=None, num_processes=None):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("🔬 Starting simulation with the following parameters:",
          f"simulation_setup_data={simulation_setup_data}, "
          f"candidate_csv={candidate_csv}, "
          f"candidate_name={candidate_name}, "
          f"candidate_bio={candidate_bio}, "
          f"config={config}")
    # 1) Run single-LLM evaluation first:
    if candidate_csv:
        print("🖋️  Running single-LLM evaluation on", candidate_csv)
        # this will produce single_llm_evaluation_results.csv in cwd
        single_llm_path = generate_response_from_sample(os.path.basename(candidate_csv))
    else:
        # if you're doing a one-off candidate_name/bio run, you could call
        # your single-LLM logic directly here instead.
        raise ValueError("single-LLM evaluation requires a candidate_csv")

    with open(simulation_setup_data, "r", encoding="utf-8") as f:
        simulation_setup_data = json.load(f)
    job_title = simulation_setup_data["job_title"]
    job_description = simulation_setup_data["job_description"]
    advisors = [{"title": x} for x in simulation_setup_data['technical_advisors']]

    if candidate_csv:
        input_data = pd.read_csv(candidate_csv).to_dict('records')
    elif candidate_name and candidate_bio:
        input_data = [{"candidate_name": candidate_name, "resume": candidate_bio}]
    else:
        raise ValueError("Either candidate_csv or both candidate_name and candidate_bio must be provided.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_files/{timestamp}"
    sentiment_agents = []
    # pbar = tqdm(input_data, desc="🔬 Simulations", unit="cand")
    for advisor_data in input_data:
        candidate_name = advisor_data["candidate_name"]
        candidate_bio = advisor_data['resume']
        tools = []
        with get_openai_callback() as cb:
            if config is not None:
                output, sentiment_agent, agents, history, initial_conditions = simulate(
                    candidate_name, candidate_bio, job_title, job_description, tools, advisors, config
                )
            else:
                output, sentiment_agent, agents, history, initial_conditions = simulate(
                    candidate_name, candidate_bio, job_title, job_description, tools, advisors
                )
        sentiment_agents.append(sentiment_agent)
        simulation_data = get_simulation_output(agents, sentiment_agent, history, output)
        costs = {
            "Total_Tokens": f"{cb.total_tokens}",
            "Prompt_Tokens": f"{cb.prompt_tokens}",
            "Completion_Tokens": f"{cb.completion_tokens}",
            "Total_Cost_USD": f"${cb.total_cost}"
        }
        simulation_data["costs"] = costs
        simulation_data["initial_conditions"] = initial_conditions

        candidate_dir = os.path.join(output_dir, candidate_name)

        if not os.path.exists(candidate_dir):
            os.makedirs(candidate_dir)

        sim_data_file = os.path.join(candidate_dir, "simulation_data.json")
        with open(sim_data_file, "w", encoding="utf-8") as f:
            json.dump(simulation_data, f, indent=2)

    print("🔬 All simulations complete. Kicking off evaluation…")
    eval_main(
        output_dir,
        resume_file=single_llm_path,
        num_processes=num_processes if num_processes is not None else max(1, mp.cpu_count() - 1)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulation for candidate(s).')
    parser.add_argument(
        '--simulation_setup_data',
        type=str,
        required=True,
        help='JSON file containing simulation setup data.'
    )
    parser.add_argument(
        '--candidate_csv',
        type=str,
        help='CSV file containing candidate names and resumes.'
    )
    parser.add_argument(
        '--candidate_name',
        type=str,
        help='Name of a single candidate.'
    )
    parser.add_argument(
        '--candidate_bio',
        type=str,
        help='Resume of a single candidate.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '-n', '--num_processes',
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help='Number of worker processes to use during evaluation.'
    )

    args = parser.parse_args()

    main(
        simulation_setup_data=args.simulation_setup_data,
        candidate_csv=args.candidate_csv,
        candidate_name=args.candidate_name,
        candidate_bio=args.candidate_bio,
        config=config,
        num_processes=args.num_processes
    )