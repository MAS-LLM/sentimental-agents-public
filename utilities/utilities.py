# Standard Library Imports
from typing import List, Dict, Any
import os
# 2025-09-19: 
# Commented out for now in case the team wants to make both OpenAI and local LLM options availabe
# from langchain.callbacks import get_openai_callback
# from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage
# 2025-09-19: 
# Commented out for now in case the team wants to make both OpenAI and local LLM options availabe
# from langchain.callbacks import get_openai_callback
# OPENAI_MODEL = os.getenv("OPENAI_MODEL") # nToDo
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_MODEL = "gpt-oss:20b"

def handle_error(error: Exception) -> str:
    """Handle errors and return a truncated message.

    Parameters:
        error (Exception): The Exception object.

    Returns:
        str: Truncated error message.
    """
    return str(error)[:50]

def generate_content_from_template(name: str, template: str, word_limit: int = None, extra_vars: Dict[str, Any] = None) -> str:
    """Generate content using a specified template. (e.g. [repo-root]/single_llm_control/advisor_prompt_template.py)

    Parameters:
        name (str): Name of the agent.
        template (str): The template to be filled.
        word_limit (int): Limit for word count.
        extra_vars (Dict[str, Any]): Extra variables to be used in formatting.

    Returns:
        str: Generated content.
    """
    prompt_vars = {'name': name, 'word_limit': word_limit}
    if extra_vars:
        prompt_vars.update(extra_vars)

    prompt = [
        HumanMessage(
            content=template.format(**prompt_vars)
        ),
    ]
    # return ChatOpenAI(model_name=OPENAI_MODEL, temperature=1.0)(prompt).content
    # ToDo - incomplete
    OLLAMA_MODEL = "gpt-oss:20b"
    return ChatOllama(model=OLLAMA_MODEL, temperature=1.0)(prompt).content

