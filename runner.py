from typing import List, Dict, Callable, Tuple, Any, Union
import os
import logging

from dotenv import load_dotenv
load_dotenv()
import concurrent.futures

from core.dialog import DialogueSimulator, DialogueAgent, DialogueAgentWithTools

from utilities import summarise_document
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
from utilities.utilities import summarise_document
from utilities.plots import create_change_sentiment_plot


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


OPENAI_MODEL = os.getenv("OPENAI_MODEL")


def run_simulation(agents: List[DialogueAgent], specified_topic: str, candidate_name, config = None) -> Tuple[str, Dict[str, Any]]:
    """Run the simulation and return the summary and analytics."""
    
    def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
        """Function to determine the next speaker."""
        return step % len(agents)
    
    if config is None:
        non_bayesian_agent = NonBayesianSentimentAgent(agents)
        nB_stopper = False
    else:
        non_bayesian_agent = NonBayesianSentimentAgent(agents, config["nonBayes_alpha"], config["nonBayes_tolerance"])
        nB_stopper = True

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
            if nB_stopper:
                break
        else:
            #wandb.log({
            #    f"{name}_sentiment_change":non_bayesian_agent.agent_tracker[name][-1],
            #    f"{name}_sentiment_value":non_bayesian_agent.change_tracker[name][-1]
            #})
            pass

    # Post-process conversation for analytics
    history = simulator.conversation_history

    summary = summarise_document(history)


    output = {
        "Candidate Name": candidate_name,
        "Summary": summary,
    }
    
    return output, non_bayesian_agent, history


def get_agent_data(job_title, job_description, candidate_df, advisors, nB_stopper, config = None):
    tools = []
    agent_names = [x.strip() for x in advisors.split(",")]
    
    agent_descriptions, agent_priorities, agent_criteria = generate_agent_information(agent_names, job_title)
    f_output = dict()
    for candidate_name, candidate_bio in zip(candidate_df["candidate_name"], candidate_df["resume"]):
        topic = generate_topic(candidate_name, candidate_bio, job_title, job_description)
        conversation_description = f"""Here is the topic of conversation: {topic}
        The participants are: {', '.join(agent_names.keys())}"""
        
        agent_system_messages = generate_system_messages(agent_names, agent_descriptions, agent_priorities, agent_criteria, tools, conversation_description)

        specified_topic = specify_topic(topic, agent_names)
        agents = initialize_agents(agent_names, agent_system_messages)

        output, non_bayesian_agent, history = run_simulation(agents, specified_topic, candidate_name, config = None, nB_stopper = nB_stopper)


        non_bayesian_data = {
            "change": non_bayesian_agent.change_tracker,
            "sentiment_data": non_bayesian_agent.agent_tracker,
        }
        fig = create_change_sentiment_plot(non_bayesian_data)
        f_output[candidate_name] = {
            "topic":topic,
            "output": output,
            "history": history,
            "non_bayesian_data": non_bayesian_data,
            "fig": fig,
            "summary": output["Summary"]
        }
    return f_output

