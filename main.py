from typing import List, Dict, Tuple, Any
import multiprocessing as mp
import os
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent / "metrics"))
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from core.dialog import DialogueSimulator, DialogueAgent
import pandas as pd
import json
from core.simulation_utilities import (
    generate_agent_information,
    generate_system_messages,
    generate_topic,
    specify_topic,
    initialize_agents,
)
from core.sentiment_agent import SentimentAgent
import argparse
import datetime
import random
from metrics.evaluation import eval_main
import warnings
import logging
import torch
from multiprocessing import get_context

# Suppress warnings and noisy logs
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

for noisy in ["pydantic", "langchain", "urllib3", "httpx"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
logging.getLogger("torch").setLevel(logging.ERROR)


class Config:
    def __init__(self,
                 dialog_temp=0.0,
                 specifyTopic_temp=0.0,
                 generateContent_temp=0.0,
                 summarize_temp=0.0,
                 model_name="llama3",
                 max_rounds=10,
                 seeds=None):
        self.dialog_temp = dialog_temp
        self.specifyTopic_temp = specifyTopic_temp
        self.generateContent_temp = generateContent_temp
        self.summarize_temp = summarize_temp
        self.max_rounds = max_rounds
        self.seeds = seeds if seeds else [42]
        self.model_name = model_name

    def __str__(self):
        return (
            f"Config(\n"
            f"    dialog_temp={self.dialog_temp},\n"
            f"    specifyTopic_temp={self.specifyTopic_temp},\n"
            f"    generateContent_temp={self.generateContent_temp},\n"
            f"    summarize_temp={self.summarize_temp},\n"
            f"    max_rounds={self.max_rounds},\n"
            f"    seeds={self.seeds},\n"
            f"    model_name={self.model_name}\n"
            f")"
        )

    def to_dict(self):
        return {
            "dialog_temp": self.dialog_temp,
            "specifyTopic_temp": self.specifyTopic_temp,
            "generateContent_temp": self.generateContent_temp,
            "summarize_temp": self.summarize_temp,
            "max_rounds": self.max_rounds,
            "seeds": self.seeds,
            "model_name": self.model_name,
        }


def set_global_seed(seed: int):
    """Ensure reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seed set to {seed}")


def _worker_init(seed: int):
    """Initializer for each worker process in multiprocessing."""
    set_global_seed(seed)
    print(f"Worker initialized with seed {seed}")


def run_simulation(
    agents: List[DialogueAgent],
    specified_topic: str,
    candidate_name,
    config=None,
    resume_context="",
    feedback_mode="none",
):
    sentiment_agent = SentimentAgent(agents)
    sentiment_agent.set_resume_context(resume_context)

    def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
        return step % len(agents)

    simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
    simulator.reset()
    simulator.inject("Facilitator", specified_topic)

    round_counter = 0
    while round_counter < config.max_rounds:
        for _ in range(len(agents)):
            name, message_content, speaker_idx = simulator.step()  # Now returns string content
            if speaker_idx is None:
                print("Skipping sentiment update because step() failed")
                continue
            sentiment_agent.update(speaker_idx)

        sentiment_agent.finalize_round(feedback_mode=feedback_mode)
        round_counter += 1

        if round_counter < config.max_rounds:
            stop = sentiment_agent.check_stopping_semantic()
            if stop:
                print(f"Stopping early after {round_counter} rounds (reason=semantic_repetition)")
                break
    else:
        print(f"Reached maximum of {config.max_rounds} rounds.")

    return sentiment_agent, simulator.conversation_history


def fetch_agent_profiles(advisors: List[str], job_title: str, config) -> str:
    agent_profiles = []

    agent_descriptions, agent_priorities, agent_criteria = generate_agent_information(
        {advisor: None for advisor in advisors},
        job_title,
        model_name=config.model_name,
        temperature=config.generateContent_temp,
    )

    for advisor in advisors:
        agent_profiles.append({
            "Agent Name": advisor,
            "Job Title": "Technical Advisor",
            "Description": agent_descriptions[advisor],
            "Priorities": agent_priorities[advisor],
            "Criteria": agent_criteria[advisor],
        })

    return json.dumps(agent_profiles, indent=2)


def simulate(
    candidate_name: str,
    candidate_bio: str,
    job_title: str,
    job_description: str,
    tools: Dict,
    advisors: List[Dict],
    config=None,
    feedback_mode: str = "none",
) -> Tuple[str, Dict[str, Any]]:
    logging.info(f"Starting simulation for {candidate_name} with feedback_mode={feedback_mode}")

    agent_names = {advisor["title"]: tools for advisor in advisors}

    # Generate agent content
    agent_descriptions, agent_priorities, agent_criteria = generate_agent_information(
        agent_names,
        job_title,
        model_name=config.model_name,
        temperature=config.generateContent_temp,
    )

    topic = generate_topic(candidate_name, candidate_bio, job_title, job_description)
    conversation_description = (
        f"Here is the topic of conversation: {topic}\n"
        f"The participants are: {', '.join(agent_names.keys())}"
    )

    agent_system_messages = generate_system_messages(
        agent_names,
        agent_descriptions,
        agent_priorities,
        agent_criteria,
        tools,
        conversation_description,
        model_name=config.model_name,
        temperature=config.generateContent_temp,
    )

    specified_topic = specify_topic(
        topic,
        agent_names,
        model_name=config.model_name,
        temperature=config.specifyTopic_temp,
    )

    # Save initial conditions with feedback_mode
    initial_conditions = {
        "agent_names": agent_names,
        "agent_descriptions": agent_descriptions,
        "agent_priorities": agent_priorities,
        "agent_criteria": agent_criteria,
        "topic": topic,
        "agent_system_messages": {
            name: msg for name, msg in agent_system_messages.items()
        },
        "specified_topic": specified_topic,
        "candidate_name": candidate_name,
        "candidate_bio": candidate_bio,
        "job_title": job_title,
        "job_description": job_description,
        "advisors": advisors,
        "feedback_mode": feedback_mode,
    }

    # Initialize agents with feedback mode
    agents = initialize_agents(
        agent_names,
        agent_system_messages,
        model_name=config.model_name,
        temperature=config.dialog_temp,
        feedback_mode=feedback_mode,
    )

    sentiment_agent, history = run_simulation(
        agents,
        specified_topic,
        candidate_name=candidate_name,
        config=config,
        resume_context=candidate_bio,
        feedback_mode=feedback_mode,
    )

    return sentiment_agent, agents, history, initial_conditions


def get_simulation_output(agents, sentiment_agent, history, config, feedback_mode):
    # Calculate rounds from conversation history
    rounds = len(history) // len(agents) if agents else 0

    base_output = {
        "agent_data": [{
            "name": agent.name,
            "system_message": agent.system_message,
            "messages": [x.to_dict() for x in agent.messages],
        } for agent in agents],
        "raw_history": history,
        "sentiment_data": sentiment_agent.get_sentiment_dynamics_data(),
        "experiment_config": {
            "temperature": config.dialog_temp,
            "max_rounds": config.max_rounds,
            "model_name": config.model_name,
        },
        "feedback_mode": feedback_mode,
        "rounds": rounds,
    }
    return base_output


def main(
        simulation_setup_data,
        candidate_csv=None,
        candidate_name=None,
        candidate_bio=None,
        models=("llama3",),
        dialog_temps=(0.0, 0.3, 0.7),
        num_processes=None,
):
    # Define feedback modes directly
    feedback_modes = ["none", "own_sentiment", "others_sentiment"]
    # feedback_modes = ["none"]

    # Load job setup data
    with open(simulation_setup_data, "r", encoding="utf-8") as f:
        simulation_setup_data = json.load(f)

    job_title = simulation_setup_data["job_title"]
    job_description = simulation_setup_data["job_description"]
    advisors = [{"title": x} for x in simulation_setup_data["technical_advisors"]]

    # Candidate input data
    if candidate_csv:
        input_data = pd.read_csv(candidate_csv).to_dict("records")
    elif candidate_name and candidate_bio:
        input_data = [{"candidate_name": candidate_name, "resume": candidate_bio}]
    else:
        raise ValueError("Either candidate_csv or both candidate_name and candidate_bio must be provided.")

    # Ensure base output directory exists
    base_output_dir = "output_files_paper"
    os.makedirs(base_output_dir, exist_ok=True)
    # Run experiment across all conditions
    for model_name in models:
        base_config = Config(
            model_name=model_name,
            # seeds=[10, 20, 30]
            seeds=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                   130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300],
        )

        print(f"Running full experiment with model: {model_name}")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(base_output_dir, f"{timestamp}_{model_name}_full_experiment")
        os.makedirs(experiment_dir, exist_ok=True)

        # Run all combinations: candidates × temperatures × seeds × feedback_modes
        for advisor_data in input_data:
            cand_name = advisor_data["candidate_name"]
            cand_bio = advisor_data["resume"]
            tools = []

            for temp in dialog_temps:
                for seed in base_config.seeds:
                    set_global_seed(seed)

                    for feedback_mode in feedback_modes:
                        print(f"Running {cand_name}, temp={temp}, seed={seed}, feedback_mode={feedback_mode}")

                        # Create config for this specific run
                        run_config = Config(
                            dialog_temp=temp,
                            specifyTopic_temp=temp,
                            generateContent_temp=temp,
                            summarize_temp=temp,
                            model_name=model_name,
                            seeds=[seed]
                        )

                        sentiment_agent, agents, history, initial_conditions = simulate(
                            cand_name, cand_bio, job_title, job_description,
                            tools, advisors, run_config, feedback_mode=feedback_mode,
                        )

                        simulation_data = get_simulation_output(agents, sentiment_agent, history, run_config,
                                                                feedback_mode)
                        simulation_data["initial_conditions"] = {
                            **initial_conditions,
                            "candidate_name": cand_name,
                            "candidate_bio": cand_bio,
                            "job_title": job_title,
                            "job_description": job_description,
                            "temperature": temp,
                            "feedback_mode": feedback_mode,
                        }
                        simulation_data["seed"] = seed

                        # Store with feedback mode in directory structure
                        cand_dir = os.path.join(experiment_dir, cand_name, f"{feedback_mode}_temp{temp}_seed{seed}")
                        os.makedirs(cand_dir, exist_ok=True)

                        sim_data_file = os.path.join(cand_dir, "simulation_data.json")
                        with open(sim_data_file, "w", encoding="utf-8") as f:
                            json.dump(simulation_data, f, indent=2)

        # Run evaluation for entire experiment
        print("All simulations complete. Running comprehensive evaluation...")
        ctx = get_context("spawn")
        num_workers = num_processes if num_processes is not None else max(1, mp.cpu_count() - 1)

        with ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(base_config.seeds[0],)) as pool:
            eval_main(experiment_dir, num_processes=num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation for candidate(s).")
    parser.add_argument(
        "--simulation_setup_data",
        type=str,
        required=True,
        help="JSON file containing simulation setup data."
    )
    parser.add_argument(
        "--candidate_csv",
        type=str,
        help="CSV file containing candidate names and resumes."
    )
    parser.add_argument(
        "--candidate_name",
        type=str,
        help="Name of a single candidate."
    )
    parser.add_argument(
        "--candidate_bio",
        type=str,
        help="Resume of a single candidate."
    )
    parser.add_argument(
        "-n", "--num_processes",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes to use during evaluation."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3"],
        help="List of LLM models to run (space separated)."
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=[0.0],
        help="List of dialog temperatures to run (space separated)."
    )

    args = parser.parse_args()
    models = ["deepseek-r1:1.5b", "llama3.2:1b", "llama3.1:8b", "mistral:7b", "gpt-oss", "gemma3:27b"]
    # models = ["deepseek-r1:1.5b"]
    dialog_temps = [0.0, 0.3, 0.7]
    main(
        simulation_setup_data=args.simulation_setup_data,
        candidate_csv=args.candidate_csv,
        candidate_name=args.candidate_name,
        candidate_bio=args.candidate_bio,
        models=models,
        dialog_temps=dialog_temps,
        num_processes=args.num_processes,
    )