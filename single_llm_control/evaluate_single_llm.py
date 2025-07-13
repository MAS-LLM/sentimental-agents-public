import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from utilities.sentiment import SentimentAnalyzer
from llama_index.llms.openai import OpenAI
from advisor_prompt_template import SINGLE_ADVISOR_PROMPT  # ✅ This brings in your template

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(SCRIPT_DIR, "..", "data", "input", "10sample.csv")

# Load candidate CV data
df = pd.read_csv(csv_path)

# Load job config
job_data = {
    "role_to_fill": "# Head of Technology in a Recycling plant scale-up in Kenya",
    "role_description": "<Insert full job description here>"
}

# Initialize tools
llm = OpenAI(model="gpt-4o-mini")  # Same as multi-agent system
analyzer = SentimentAnalyzer()

# Prompt template
def build_prompt(candidate_name, candidate_bio, role_to_fill, role_description):
    return SINGLE_ADVISOR_PROMPT.format(
        candidate_name=candidate_name,
        candidate_bio=candidate_bio,
        role_to_fill=role_to_fill,
        role_description=role_description,
    )

# Store results
results = []

for _, row in df.iterrows():
    prompt = build_prompt(
        row["candidate_name"],
        row["resume"],
        job_data["role_to_fill"],
        job_data["role_description"]
    )

    # Call LLM
    opinion = llm.complete(prompt).text.strip()

    # Sentiment + Keyword Analysis
    sentiment_data = analyzer.analyze_message(opinion)

    results.append({
        "candidate_name": row["candidate_name"],
        "opinion": opinion,
        "overall_sentiment": sentiment_data["overall_sentiment"],
        "keywords": [kw["keyword"] for op in sentiment_data["opinions"] for kw in op["keywords"]],
    })

# Save results
pd.DataFrame(results).to_csv("single_llm_evaluation_results.csv", index=False)
