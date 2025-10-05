from typing import List, Dict, Tuple
from dotenv import load_dotenv
from core.dialog import DialogueAgent, DialogueAgentWithTools, DialogueAgentWithOwnSentimentFeedback, DialogueAgentWithOthersSentimentFeedback
# from core.advisory_brief import (
#     TOPIC, ADVISOR_PRIORITIES, ADVISOR_DESCRIPTION,
#     ADVISOR_CRITERIA, SYSTEM_MESSAGE, SPECIFIC_TOPIC,
# )
from core.advisory_brief_paper import (
    TOPIC, ADVISOR_PRIORITIES, ADVISOR_DESCRIPTION,
    ADVISOR_CRITERIA, SYSTEM_MESSAGE, SPECIFIC_TOPIC,
)
from utilities.utilities import generate_content_from_template, get_model

# Load environment variables (not strictly needed for Ollama, but kept for consistency)
load_dotenv()

# ─────────────────────────────────────────────────────────────
# Agent information generation
# ─────────────────────────────────────────────────────────────
def generate_agent_information(
    agent_names: Dict,
    job_title: str,
    model_name: str,
    temperature: float,
) -> Tuple[Dict, Dict, Dict]:
    """Generate descriptions, priorities, and criteria for agents using Ollama models."""
    word_limit = 10
    agent_descriptions = {
        name: generate_content_from_template(
            name,
            ADVISOR_DESCRIPTION,
            word_limit,
            model_name=model_name,
            temperature=temperature,
        )
        for name in agent_names
    }
    agent_priorities = {
        name: generate_content_from_template(
            name,
            ADVISOR_PRIORITIES,
            word_limit,
            model_name=model_name,
            temperature=temperature,
        )
        for name in agent_names
    }
    agent_criteria = {
        name: generate_content_from_template(
            name,
            ADVISOR_CRITERIA,
            word_limit,
            extra_vars={"role_to_fill": job_title},
            model_name=model_name,
            temperature=temperature,
        )
        for name in agent_names
    }
    return agent_descriptions, agent_priorities, agent_criteria


def generate_topic(candidate_name: str, candidate_bio: str, job_title: str, job_description: str) -> str:
    """Generate the topic of the conversation from static template."""
    return TOPIC.format(
        candidate_name=candidate_name,
        candidate_bio=candidate_bio,
        role_to_fill=job_title,
        role_description=job_description,
    )

# In simulation_utilities.py, add debugging to system message generation
def generate_system_messages(
        agent_names: Dict,
        agent_descriptions: Dict,
        agent_priorities: Dict,
        agent_criteria: Dict,
        tools: Dict,
        conversation_description: str,
        model_name: str,
        temperature: float,
) -> Dict[str, str]:
    """Generate system messages for each agent as direct role instructions.
    
    CRITICAL: Does NOT use LLM generation. Constructs messages directly from components
    to avoid creating biographical narratives that cause meta-commentary.
    """
    system_messages = {}

    for (name, tools), description, priority, criterion in zip(
            agent_names.items(),
            agent_descriptions.values(),
            agent_priorities.values(),
            agent_criteria.values(),
    ):
        # DIRECT CONSTRUCTION - No LLM call
        system_msg = (
            f"You are {name}.\n\n"
            f"Role: {description.strip()}\n\n"
            f"Your priorities: {priority.strip()}\n\n"
            f"Evaluation criteria: {criterion.strip()}\n\n"
            f"Task: {conversation_description.strip()}\n\n"
            f"Provide your evaluation based on the criteria above. "
            f"Respond in English only. Do not speak from the perspective of other participants."
        )
        
        system_messages[name] = system_msg

    return system_messages


def specify_topic(
    topic: str,
    agent_names: Dict,
    model_name: str,
    temperature: float,
) -> str:
    """Make the topic more specific using Ollama model."""
    prompt = (
        f"Based on this evaluation task:\n{topic}\n\n"
        f"Provide a clear, structured evaluation prompt (50words) that tells "
        f"the participants ({', '.join(agent_names)}) what specific aspects they should assess. "
        f"The prompt should be actionable and guide them to provide independent, detailed evaluations."
    )
    # print(f"\n[SPECIFY_TOPIC] Generating structured evaluation prompt...")
    # print(f"[SPECIFY_TOPIC_INPUT] Original topic: {topic}")

    llm = get_model(model_name, temperature)
    response = llm.invoke(prompt)   # Ollama returns plain string
    return response.strip()


# ─────────────────────────────────────────────────────────────
# Agent initialization
# ─────────────────────────────────────────────────────────────
def initialize_agents(
        agent_names: Dict,
        agent_system_messages: Dict[str, str],
        model_name: str,
        temperature: float,
        feedback_mode: str = "none",  # "none", "own_sentiment", "others_sentiment"
) -> List[DialogueAgent]:
    """Initialize agents based on sentiment feedback mode."""
     
    print("\n=== AGENT INITIALIZATION ORDER ===")
    for idx, (name, tools) in enumerate(agent_names.items()):
        print(f"Position {idx}: {name}")
    print("===================================\n")

    if feedback_mode == "own_sentiment":
        return [
            DialogueAgentWithOwnSentimentFeedback(
                name=name,
                system_message=system_message,
                model_name=model_name,
                tools=tools,
                temperature=temperature,
            )
            for (name, tools), system_message in zip(
                agent_names.items(), agent_system_messages.values()
            )
        ]
    elif feedback_mode == "others_sentiment":
        return [
            DialogueAgentWithOthersSentimentFeedback(
                name=name,
                system_message=system_message,
                model_name=model_name,
                tools=tools,
                temperature=temperature,
            )
            for (name, tools), system_message in zip(
                agent_names.items(), agent_system_messages.values()
            )
        ]
    else:  # feedback_mode == "none"
        return [
            DialogueAgentWithTools(
                name=name,
                system_message=system_message,
                model_name=model_name,
                tools=tools,
                temperature=temperature,
            )
            for (name, tools), system_message in zip(
                agent_names.items(), agent_system_messages.values()
            )
        ]