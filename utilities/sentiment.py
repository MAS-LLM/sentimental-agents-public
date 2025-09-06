# utilities/sentiment.py
import torch
from collections import defaultdict
from transformers import pipeline

class SentimentAnalyzer:
    """Analyze sentiment directly on a message (no extra LLM structuring)."""

    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_code = 0 if device == "cuda:0" else -1

        # Hugging Face sentiment classifier (defaults to SST-2)
        self.classifier = pipeline(
            "text-classification",
            return_all_scores=True,
            device=device_code
        )

        self.initial_sentiment_score = 0
        self.sentiment_scores = defaultdict(lambda: self.initial_sentiment_score)

    def classify(self, text: str) -> float:
        """Return POSITIVE - NEGATIVE probability difference in [-1, 1]."""
        results = self.classifier(text)[0]
        pos = next((r["score"] for r in results if r["label"] == "POSITIVE"), 0.0)
        neg = next((r["score"] for r in results if r["label"] == "NEGATIVE"), 0.0)
        return pos - neg

    def analyze_message(self, message: str) -> dict:
        """
        Score the whole message directly.
        Returns:
            {
                "overall_sentiment": float,  # [-1, 1]
                "label": "positive" | "neutral" | "negative"
            }
        """
        score = self.classify(message)
        label = "neutral"
        # simple thresholding; tweak if you like
        if score > 0.05:
            label = "positive"
        elif score < -0.05:
            label = "negative"

        return {
            "overall_sentiment": score,
            "label": label
        }



# import os
# import torch
# import logging
# from typing import List
# from collections import defaultdict
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field
# from transformers import pipeline
#
# from llama_index.llms.openai import OpenAI
# from llama_index.program.openai import OpenAIPydanticProgram
#
# # Load environment variables
# load_dotenv()
#
# # Set logging configuration
# # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
#
# # Set device based on GPU availability
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device_code = 0 if device == "cuda:0" else -1
#
# # Get OpenAI model from environment variables
# OPENAI_MODEL = os.getenv("OPENAI_MODEL")
#
# # HR prompt template
# HR_PROMPT_TEMPLATE = '''
# You are an experienced HR Advisor.
# Convert the message to 5 individual points with keywords representing each point.
# Be very emotional while at it.
# Message: {message}
# '''
#
# class Keyword(BaseModel):
#     """Data model for a keyword."""
#     keyword: str
#
# class Opinion(BaseModel):
#     """Data model for an opinion."""
#     keywords: List[Keyword]
#     opinion: str
#
# class Response(BaseModel):
#     """Data model for a list of opinions."""
#     opinions: List[Opinion]
#
# class AnalyzedOpinion(BaseModel):
#     """Data model for an analyzed opinion."""
#     opinion: str
#     agents: List[str]
#
# class OpinionReport(BaseModel):
#     """Data model for a candidate report."""
#     strong_opinions: List[AnalyzedOpinion]
#     agreeable_opinions: List[AnalyzedOpinion]
#     extensively_discussed_opinions: List[AnalyzedOpinion]
#
# # Initialize classifier pipeline
# classifier = pipeline(
#     "text-classification",
#     return_all_scores=True,
#     device=device_code
# )
#
# llm = OpenAI(model=OPENAI_MODEL)
#
#
# class SentimentAnalyzer:
#     """A class to analyze sentiments from messages using a model."""
#
#     def __init__(self):
#         self.opinion_history = []
#         self.initial_sentiment_score = 0
#         self.sentiment_scores = defaultdict(lambda: self.initial_sentiment_score)
#         self.program = OpenAIPydanticProgram.from_defaults(
#             output_cls=Response,
#             llm=llm,
#             prompt_template_str=HR_PROMPT_TEMPLATE,
#             verbose=False,
#         )
#
#     def classify(self, text: str) -> float:
#         """Classify the sentiment of a given text using raw model scores."""
#         # Get all scores from the default model
#         results = classifier(text)[0]
#
#         # Find positive and negative scores
#         positive_score = 0
#         negative_score = 0
#
#         for result in results:
#             if result['label'] == 'POSITIVE':
#                 positive_score = result['score']
#             elif result['label'] == 'NEGATIVE':
#                 negative_score = result['score']
#         return positive_score - negative_score
#
#     def analyze_message(self, message: str) -> dict:
#         """Fetch opinions and sentiment scores for a given message."""
#         res = self.program(message=message)
#         # logging.info("Fetched opinions")
#         opinion_dict = res.dict()
#
#         # Get sentiment scores for each opinion using raw model scores
#         scores = [self.classify(entry["opinion"]) for entry in opinion_dict["opinions"]]
#
#         # Calculate overall sentiment as average of all opinion scores
#         overall_score = sum(scores) / len(scores) if scores else 0
#
#         # Add overall sentiment to the result
#         opinion_dict["overall_sentiment"] = overall_score
#
#         return opinion_dict