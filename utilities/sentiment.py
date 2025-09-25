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
        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"

        return {
            "overall_sentiment": score,
            "label": label
        }