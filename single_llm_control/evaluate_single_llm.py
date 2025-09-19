#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import random
import numpy as np
import datetime

from utilities.sentiment import SentimentAnalyzer
from advisor_prompt_template import (
    SINGLE_TOPIC,
    SINGLE_DESCRIPTION,
    SINGLE_PRIORITIES,
    SINGLE_CRITERIA,
)

# 2025-09-19: 
# Commented out for now in case the team wants to make both OpenAI and local LLM options availabe
# === OpenAI (official SDK) ===
# pip install openai>=1.0.0
# from openai import OpenAI
from langchain_community.chat_models import ChatOllama

# 2025-09-19: 
# Commented out for now in case the team wants to make both OpenAI and local LLM options availabe
# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # must be set
OLLAMA_MODEL = "gpt-oss:20b"

def build_prompt(candidate_name, candidate_bio, role_to_fill, role_description):
    topic = SINGLE_TOPIC.format(
        candidate_name=candidate_name,
        candidate_bio=candidate_bio,
        role_to_fill=role_to_fill,
        role_description=role_description
    )
    description = SINGLE_DESCRIPTION.format(word_limit=50)
    priorities  = SINGLE_PRIORITIES.format(word_limit=50)
    criteria    = SINGLE_CRITERIA.format(role_to_fill=role_to_fill, word_limit=80)

    return f"""{topic}

{description}

{priorities}

{criteria}

Now, as the UnifiedAdvisor, synthesize your pros vs. cons in one strong, emotionally upfront paragraph.
Base your reasoning strictly on the description, priorities, and criteria above.
"""

def generate_response_from_sample(csv_filename=None, output_dir=None):
    # --- Reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # --- Resolve CSV path
    if not csv_filename:
        raise ValueError("csv_filename must be provided.")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(SCRIPT_DIR, "..", "data", "input", csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- Job config
    job_data = {
        "role_to_fill": "# Head of Technology in a Recycling Plant Scale-Up",
        "role_description": (
            "We are a B-Corp-certified recycling initiative with 2,500+ collectors "
            "and 6,000 tons recycled annually. Seeking a strategic, hands-on Head of Technology "
            "to design modular, cloud-based infrastructure, automate supply-chain processes, "
            "and build a high-performance team. Reports to the CIO; requires 5+ years in senior "
            "tech leadership, full-stack expertise, and agile delivery experience."
        )
    }
# 2025-09-19: 
# Commented out for now in case the team wants to make both OpenAI and local LLM options availabe
    # --- OpenAI client
    # if not OPENAI_API_KEY:
    #     raise EnvironmentError("OPENAI_API_KEY is not set.")
    # client = OpenAI(api_key=OPENAI_API_KEY)
    ollama_model = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)


    analyzer = SentimentAnalyzer()
    results  = []

    for _, row in df.iterrows():
        prompt = build_prompt(
            row["candidate_name"],
            row["resume"],
            job_data["role_to_fill"],
            job_data["role_description"]
        )

# 2025-09-19: 
# Commented out for now in case the team wants to make both OpenAI and local LLM options availabe
        # try:
            # Chat Completions API (stable)
        #     chat = client.chat.completions.create(
        #         model = OLLAMA_MODEL ,
        #         temperature=0.0,
        #         max_tokens=600,
        #         messages=[
        #             {"role": "system", "content": "You are a concise, direct hiring advisor."},
        #             {"role": "user", "content": prompt},
        #         ],
        #     )
        #     opinion = (chat.choices[0].message.content or "").strip()
        # except Exception as e:
        #     print(f"Error processing {row['candidate_name']}: {e}")
        #     opinion = "Error generating response"
        try:
            chat_response = ollama_model.invoke([
                {"role": "system", "content": "You are a concise, direct hiring advisor."},
                {"role": "user", "content": prompt},
            ])
            opinion = (chat_response.content or "").strip()
        except Exception as e:
            print(f"Error processing {row['candidate_name']}: {e}")
            opinion = "Error generating response"

        sentiment_data = analyzer.analyze_message(opinion)
        results.append({
            "candidate_name": row["candidate_name"],
            "opinion": opinion,
            "overall_sentiment": sentiment_data.get("overall_sentiment"),
        })

    # --- Output path
    if output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output_files", "single_llm", ts)
    os.makedirs(output_dir, exist_ok=True)

    base     = os.path.splitext(csv_filename)[0]
    out_path = os.path.join(output_dir, f"{base}_single_llm_evaluation_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    return out_path
