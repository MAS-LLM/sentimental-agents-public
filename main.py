from typing import List, Dict, Callable, Tuple, Any, Union
import os
import logging

from dotenv import load_dotenv
load_dotenv()
import concurrent.futures

from core.dialog import DialogueSimulator, DialogueAgent, DialogueAgentWithTools

from utilities.utilities import summarise_document
#from tool_loader import TOOLS
from utilities.data_loader import load_file
import pandas as pd

from collections import defaultdict
import json
import wandb


from core.simulation_utilities import generate_agent_information, generate_system_messages, generate_topic, specify_topic, initialize_agents
from core.non_bayesian import NonBayesianSentimentAgent
from langchain.callbacks import get_openai_callback
from metrics.decision_making import DecisionMaker
from utilities.opinion_analyser import AdvisorReport
import argparse
import pandas as pd
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


OPENAI_MODEL = os.getenv("OPENAI_MODEL")




class Config:
    def __init__(self):
        self.dialog_temp = 1.5
        self.specifyTopic_temp = 1.5
        self.generateContent_temp = 1 # Content gen for agent messages TODO: split parameters for individual agents
        self.summarize_temp = 0
        self.nonBayes_alpha = 0.5
        self.nonBayes_tolerance = 0
        self.max_rounds = 3

    def __str__(self):
        return (f"Config(\n"
                f"    dialog_temp={self.dialog_temp},\n"
                f"    specifyTopic_temp={self.specifyTopic_temp},\n"
                f"    generateContent_temp={self.generateContent_temp},\n"
                f"    summarize_temp={self.summarize_temp},\n"
                f"    nonBayes_alpha={self.nonBayes_alpha},\n"
                f"    nonBayes_tolerance={self.nonBayes_tolerance},\n"
                f"    max_rounds={self.max_rounds}\n"
                f")")

    def to_dict(self):
        return {
            "dialog_temp": self.dialog_temp,
            "specifyTopic_temp": self.specifyTopic_temp,
            "generateContent_temp": self.generateContent_temp,
            "summarize_temp": self.summarize_temp,
            "nonBayes_alpha": self.nonBayes_alpha,
            "nonBayes_tolerance": self.nonBayes_tolerance,
            "max_rounds": self.max_rounds
        }

# Usage:
config = Config()
#print(config)
#print(config.to_dict())


def run_simulation(agents: List[DialogueAgent], specified_topic: str, candidate_name, config = None) -> Tuple[str, Dict[str, Any]]:
    """Run the simulation and return the summary and analytics."""
    
    def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
        """Function to determine the next speaker."""
        return step % len(agents)
    
    if config is None:
        non_bayesian_agent = NonBayesianSentimentAgent(agents)
    else:
        non_bayesian_agent = NonBayesianSentimentAgent(agents, config.nonBayes_alpha, config.nonBayes_tolerance)

    simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
    simulator.reset()
    simulator.inject("Facilitator", specified_topic)
    
    
    # Main dialogue loop
    if config is None:
        max_iters = 3 * len(agents)
    else:
        max_iters = config.max_rounds * len(agents)
    for i in range(max_iters):
        message = None
        name, agent_message, speaker_idx = simulator.step()  # Get new data from the simulator
        if non_bayesian_agent.update(speaker_idx) == "Break":
            break
        else:
            #wandb.log({
            #    f"{name}_sentiment_change":non_bayesian_agent.agent_tracker[name][-1],
            #    f"{name}_sentiment_value":non_bayesian_agent.change_tracker[name][-1]
            #})
            pass

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
    
    return output, non_bayesian_agent, history

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
        config = None
    ) -> Tuple[str, Dict[str, Any]]:
    
    logging.info("Starting the simulation.")
    
    agent_names = {advisor["title"]: tools for advisor in advisors}
    
    agent_descriptions, agent_priorities, agent_criteria = generate_agent_information(agent_names, job_title)
    
    topic = generate_topic(candidate_name, candidate_bio, job_title, job_description)
    conversation_description = f"""Here is the topic of conversation: {topic}
    The participants are: {', '.join(agent_names.keys())}"""
    
    agent_system_messages = generate_system_messages(agent_names, agent_descriptions, agent_priorities, agent_criteria, tools, conversation_description)
    
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
        agents = initialize_agents(agent_names, agent_system_messages, temperature = config.dialog_temp)
    else:
        agents = initialize_agents(agent_names, agent_system_messages)
    output, non_bayesian_agent, history  = run_simulation(agents, specified_topic, candidate_name = candidate_name, config = config)
    
    return output, non_bayesian_agent, agents, history, initial_conditions



def get_simulation_output(agents, non_bayesian_agent, history, output):
    #dm = DecisionMaker(agents)
    #decision_metrics = {
    #    x.name: x.decision_metrics for x in dm.agents
    #}
    report = AdvisorReport(agents)
    agent_data = [{
        "name":agent.name,
        "messages": [x.to_dict() for x in agent.messages],
    } for agent in agents]
    out = {
        "agent_data":agent_data,
        "raw_history":history,
        "summarized_output":output,
        "opinion_report":report.generate().to_dict(orient = "records"),
        #"decision_metrics": decision_metrics,
        "non_bayesian_data":{
            "change": non_bayesian_agent.change_tracker,
            "sentiment_data": non_bayesian_agent.agent_tracker,
        }
    }
    return out


def main(simulation_setup_data, candidate_csv=None, candidate_name=None, candidate_bio=None, config = None):
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
    non_bayesian_agents = []
    for advisor_data in input_data:
        candidate_name = advisor_data["candidate_name"]
        candidate_bio = advisor_data['resume']
        tools = []
        with get_openai_callback() as cb:
            if config is not None:
                output, non_bayesian_agent, agents, history, initial_conditions = simulate(
                    candidate_name, candidate_bio, job_title, job_description, tools, advisors, config
                )
            else:
                output, non_bayesian_agent, agents, history, initial_conditions = simulate(
                    candidate_name, candidate_bio, job_title, job_description, tools, advisors
                )
        non_bayesian_agents.append(non_bayesian_agent)
        simulation_data = get_simulation_output(agents, non_bayesian_agent, history, output)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulation for candidate(s).')
    parser.add_argument('--simulation_setup_data', type=str, help='JSON file containing simulation setup data.')
    parser.add_argument('--candidate_csv', type=str, help='CSV file containing candidate names and resumes.')
    parser.add_argument('--candidate_name', type=str, help='Name of a single candidate.')
    parser.add_argument('--candidate_bio', type=str, help='Resume of a single candidate.')
    args = parser.parse_args()
    main(args.simulation_setup_data, args.candidate_csv, args.candidate_name, args.candidate_bio, config = config)
