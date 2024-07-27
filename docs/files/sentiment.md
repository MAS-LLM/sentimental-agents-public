# Sentiment Analysis Documentation.

## Overview

The `sentiment.py` file contains the implementation of a sentiment analyzer class (`SentimentAnalyzer`) utilizing a combination of transformers and OpenAI models. The primary purpose is to analyze sentiments and opinions from messages, providing sentiment scores and opinions based on a predefined template.

## Environment Setup

The file starts by loading environment variables using `dotenv` and configuring the logging level. It then determines the device (GPU or CPU) based on availability and initializes the OpenAI model.

## Data Models

The file defines several Pydantic data models to structure the output of sentiment analysis:

- **Keyword**: Represents a keyword extracted from a message.
- **Opinion**: Models an opinion consisting of a list of keywords and the opinion text itself.
- **Response**: Represents a list of opinions.
- **AnalyzedOpinion**: Describes an analyzed opinion, including the opinion text and associated agents.
- **OpinionReport**: Models a complete candidate report, categorizing opinions into strong, agreeable, and extensively discussed.

## SentimentAnalyzer Class

The core of the file is the `SentimentAnalyzer` class, responsible for sentiment analysis and opinion extraction. The class initializes with an OpenAI model, a sentiment score history, and a predefined sentiment analysis program.

### Methods

- **`classify(self, text: str) -> float`**: Classifies the sentiment of a given text using a transformer-based text classification pipeline.
- **`analyze_message(self, message: str) -> dict`**: Fetches opinions and sentiment scores for a given message using the OpenAI model and sentiment classifier.

## HR Prompt Template

The file includes an HR prompt template (`HR_PROMPT_TEMPLATE`) used in sentiment analysis. This template instructs the model to convert a message into five individual points with emotional keywords representing each point.

## External Dependencies

The sentiment analyzer uses the `transformers` library for the text classification pipeline and relies on the `pipeline` method to load the CouchCat model. Additionally, it utilizes the `llama_index` library for handling OpenAI models.

## Usage

The `sentiment.py` file can be used within the broader context of a conversation or message analysis system. The `SentimentAnalyzer` class is instantiated to analyze sentiments and opinions from messages, providing valuable insights into the emotional content and key points expressed.

## Note

Ensure that the required dependencies and environment variables, especially the OpenAI model, are properly configured for the sentiment analysis to function effectively.