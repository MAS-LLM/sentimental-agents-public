#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import random
import numpy as np

from utilities.sentiment import SentimentAnalyzer
from llama_index.llms.openai import OpenAI
from advisor_prompt_template import (
    SINGLE_TOPIC,
    SINGLE_DESCRIPTION,
    SINGLE_PRIORITIES,
    SINGLE_CRITERIA,
)

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
Give a clear overall_sentiment in [-1, +1] at the end."""

def generate_response_from_sample(csv_filename=None, output_dir=None):
    # reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(SCRIPT_DIR, "..", "data", "input", csv_filename)
    df         = pd.read_csv(csv_path)

    # Full job configuration
    # In your single‐agent script
    job_data = {
        "role_to_fill": "# Head of Technology in a Recycling Plant Scale‑Up",
        "role_description": (
            "We are a B‑Corp‑certified recycling initiative with 2,500+ collectors "
            "and 6,000 tons recycled annually.  Seeking a strategic, hands‑on Head of Technology "
            "to design modular, cloud‑based infrastructure, automate supply‑chain processes, "
            "and build a high‑performance team.  Reports to the CIO; requires 5+ years in senior "
            "tech leadership, full‑stack expertise, and agile delivery experience."
        )
    }

    llm      = OpenAI(model="gpt-4o-mini", temperature=0.0)
    analyzer = SentimentAnalyzer()
    results  = []

    for _, row in df.iterrows():
        prompt = build_prompt(
            row["candidate_name"],
            row["resume"],
            job_data["role_to_fill"],
            job_data["role_description"]
        )
        opinion = llm.complete(prompt).text.strip()
        sentiment_data = analyzer.analyze_message(opinion)

        results.append({
            "candidate_name":    row["candidate_name"],
            "opinion":           opinion,
            "overall_sentiment": sentiment_data["overall_sentiment"],
        })

    os.makedirs(output_dir, exist_ok=True)
    base     = os.path.splitext(csv_filename)[0]
    out_path = os.path.join(output_dir, f"{base}_single_llm_evaluation_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)

    return out_path
