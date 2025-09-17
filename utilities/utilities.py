# utilities.py

import os
from typing import Dict, Any
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage

# Define the available Ollama models you’ve pulled locally
OLLAMA_MODELS = {
    "llama3": "llama3",
    "mistral": "mistral",
    "gemma": "gemma",
    "gpt-oss": "gpt-oss",
    # Add more if you have pulled them via `ollama pull <model>`
}


def get_model(model_name: str = "llama3", temperature: float = 0.3):
    """Return an Ollama-backed model via LangChain OllamaLLM."""
    if model_name not in OLLAMA_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(OLLAMA_MODELS.keys())}")

    model_id = OLLAMA_MODELS[model_name]

    return OllamaLLM(
        model=model_id,
        temperature=temperature,
        # optional extras you can pass:
        num_ctx=2048,   # context window size
        num_predict=200 # max tokens to generate
    )


def handle_error(error: Exception) -> str:
    """Truncate error message for logging/agents."""
    return str(error)[:50]


def generate_content_from_template(
    name: str,
    template: str,
    word_limit: int = None,
    extra_vars: Dict[str, Any] = None,
    model_name: str = "llama3",
    temperature: float = 0.3,
) -> str:
    """Generate content by filling a template and running it through the selected chat model."""
    prompt_vars = {"name": name, "word_limit": word_limit}
    if extra_vars:
        prompt_vars.update(extra_vars)

    prompt_text = template.format(**prompt_vars)
    chat_model = get_model(model_name, temperature)

    # OllamaLLM works with plain strings (no need to wrap in HumanMessage)
    response = chat_model.invoke(prompt_text)
    return response if isinstance(response, str) else str(response)
