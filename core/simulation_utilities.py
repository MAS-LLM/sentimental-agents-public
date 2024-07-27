from typing import List, Dict, Callable, Tuple, Any, Union
import os
import logging

from dotenv import load_dotenv
load_dotenv()

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from core.dialog import DialogueSimulator, DialogueAgent, DialogueAgentWithTools
#from sentiment import classify_text
from core.advisory_brief import (
    TOPIC, ADVISOR_PRIORITIES, ADVISOR_DESCRIPTION, 
    ADVISOR_CRITERIA, SYSTEM_MESSAGE, SPECIFIC_TOPIC,
)

from utilities.utilities import generate_content_from_template

OPENAI_MODEL = os.getenv("OPENAI_MODEL")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def generate_agent_information(agent_names: Dict, job_title: str) -> Tuple[Dict, Dict, Dict]:
    """Generate descriptions, priorities, and criteria for agents."""
    word_limit = 10  # word limit for task brainstorming between agents
    agent_descriptions = {name: generate_content_from_template(name, ADVISOR_DESCRIPTION, word_limit) for name in agent_names}
    agent_priorities = {name: generate_content_from_template(name, ADVISOR_PRIORITIES, word_limit) for name in agent_names}
    agent_criteria = {name: generate_content_from_template(name, ADVISOR_CRITERIA, word_limit, extra_vars={"role_to_fill": job_title}) for name in agent_names}
    
    return agent_descriptions, agent_priorities, agent_criteria

def generate_topic(candidate_name: str, candidate_bio: str, job_title: str, job_description: str) -> str:
    """Generate the topic of the conversation."""
    return TOPIC.format(candidate_name=candidate_name, candidate_bio=candidate_bio, role_to_fill=job_title, role_description=job_description)

def generate_system_messages(agent_names: Dict, agent_descriptions: Dict, agent_priorities: Dict, agent_criteria: Dict, tools: Dict, conversation_description: str) -> Dict:
    """Generate system messages for each agent."""
    return {
        name: generate_content_from_template(
            name, SYSTEM_MESSAGE, 
            extra_vars={
                "description": description,
                "priority": priority,
                "criterion": criterion,
                "tools": tools,
                "conversation_description": conversation_description
            }
        ) for (name, tools), description, priority, criterion in zip(
            agent_names.items(), agent_descriptions.values(), agent_priorities.values(), agent_criteria.values()
        )
    }



def specify_topic(topic: str, agent_names: Dict, temperature = 1.5) -> str:
    """Make the topic more specific."""
    topic_specifier_prompt = [
        SystemMessage(content="You can make a topic more specific."),
        HumanMessage(content=SPECIFIC_TOPIC.format(topic=topic, word_limit=5, names=', '.join(agent_names)))
    ]
    return ChatOpenAI(model_name=OPENAI_MODEL, temperature=temperature)(topic_specifier_prompt).content #change temperature from 1.0 to 0.7

def initialize_agents(agent_names: Dict, agent_system_messages: Dict, temperature = 1.5) -> List[DialogueAgent]:
    """Initialize agents for the conversation."""
    return [
        DialogueAgentWithTools(
            name=name,
            system_message=SystemMessage(content=system_message),
            model=ChatOpenAI(model_name=OPENAI_MODEL, temperature=temperature), 
            tools=tools,
            top_k_results=2
        ) for (name, tools), system_message in zip(agent_names.items(), agent_system_messages.values())
    ]