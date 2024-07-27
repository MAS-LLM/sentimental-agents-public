import os
import torch
import logging
from typing import List
from collections import defaultdict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from transformers import pipeline

from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram

# Load environment variables
load_dotenv()

# Set logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Set device based on GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_code = 0 if device == "cuda:0" else -1

# Get OpenAI model from environment variables
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# HR prompt template
HR_PROMPT_TEMPLATE = '''
You are an experienced HR Advisor.
Convert the message to 5 individual points with keywords representing each point.
Be very emotional while at it.
Message: {message}
'''

class Keyword(BaseModel):
    """Data model for a keyword."""
    keyword: str

class Opinion(BaseModel):
    """Data model for an opinion."""
    keywords: List[Keyword]
    opinion: str

class Response(BaseModel):
    """Data model for a list of opinions."""
    opinions: List[Opinion]

class AnalyzedOpinion(BaseModel):
    """Data model for an analyzed opinion."""
    opinion: str
    agents: List[str]

class OpinionReport(BaseModel):
    """Data model for a candidate report."""
    strong_opinions: List[AnalyzedOpinion]
    agreeable_opinions: List[AnalyzedOpinion]
    extensively_discussed_opinions: List[AnalyzedOpinion]

# Initialize classifier pipeline
classifier = pipeline(
    "text-classification",
    model="CouchCat/ma_sa_v7_distil",
    return_all_scores=True,
    device=device_code
)

llm = OpenAI(model=OPENAI_MODEL)

class SentimentAnalyzer:
    """A class to analyze sentiments from messages using a model."""

    def __init__(self):
        self.opinion_history = []
        self.initial_sentiment_score = 0
        self.sentiment_scores = defaultdict(lambda: self.initial_sentiment_score)
        self.program = OpenAIPydanticProgram.from_defaults(
            output_cls=Response,
            llm=llm,
            prompt_template_str=HR_PROMPT_TEMPLATE,
            verbose=False,
        )

    def classify(self, text: str) -> float:
        """Classify the sentiment of a given text."""
        score = classifier(text)[0][-1]["score"]
        return 2 * score - 1  # Rescale value from [0,1] to [-1,1]

    def analyze_message(self, message: str) -> dict:
        """Fetch opinions and sentiment scores for a given message."""
        res = self.program(message=message)
        logging.info("Fetched opinions")
        opinion_dict = res.dict()
        scores = [self.classify(entry["opinion"]) for entry in opinion_dict["opinions"]]
        overall_score = sum(scores) / len(scores)
        opinion_dict["overall_sentiment"] = overall_score
        return opinion_dict


