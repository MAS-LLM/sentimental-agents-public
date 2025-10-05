import torch
from collections import defaultdict
from transformers import pipeline

class SentimentAnalyzer:
    """Analyze sentiment directly on a message (no extra LLM structuring)."""
    
    def __init__(self, force_cpu=False):
        # Check GPU availability but allow override
        if force_cpu:
            device_code = -1
            print("SentimentAnalyzer: Forcing CPU mode")
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            device_code = 0 if device == "cuda:0" else -1
            print(f"SentimentAnalyzer: Using device {device}")
        
        # Explicitly specify model to avoid the warning
        self.classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",  # ← ADD THIS
            return_all_scores=True,
            device=device_code,
            truncation=True,
            max_length=512
        )
        
        self.initial_sentiment_score = 0
        self.sentiment_scores = defaultdict(lambda: self.initial_sentiment_score)

    def classify(self, text: str) -> float:
        """
        Return POSITIVE - NEGATIVE probability difference in [-1, 1].
        Handles long texts by truncating to model's max length.
        """
        if not text or not text.strip():
            return 0.0
        
        # Pre-truncate very long texts (approximate: 1 token ≈ 4 chars)
        max_chars = 512 * 4  # ~2048 characters
        if len(text) > max_chars:
            text = text[:max_chars]
        
        try:
            results = self.classifier(text)[0]
            pos = next((r["score"] for r in results if r["label"] == "POSITIVE"), 0.0)
            neg = next((r["score"] for r in results if r["label"] == "NEGATIVE"), 0.0)
            return pos - neg
        except Exception as e:
            print(f"Warning: Sentiment classification failed: {e}")
            print(f"Text length: {len(text)} chars")
            return 0.0  # Return neutral on error

    def analyze_message(self, message: str) -> dict:
        """
        Score the whole message directly.
        Returns:
        {
            "overall_sentiment": float,  # [-1, 1]
            "label": "positive" | "neutral" | "negative"
        }
        """
        if not message or not message.strip():
            return {
                "overall_sentiment": 0.0,
                "label": "neutral"
            }
        
        try:
            score = self.classify(message)
            label = "neutral"
            
            # Simple thresholding
            if score > 0.1:
                label = "positive"
            elif score < -0.1:
                label = "negative"
            
            return {
                "overall_sentiment": score,
                "label": label
            }
        except Exception as e:
            print(f"Error in analyze_message: {e}")
            return {
                "overall_sentiment": 0.0,
                "label": "neutral"
            }