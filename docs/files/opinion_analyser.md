# Opinion Analysis Documenation

## Overview

The `opinion_analyser.py` file contains the implementation of an opinion analyzer, including data models for different types of opinions and a report generation mechanism. The analyzer leverages natural language processing techniques, including sentence embeddings and polyfuzz matching, to analyze opinions expressed in messages from different advisors.

## Data Models

### StrongOpinions Class

- Represents viewpoints or beliefs held with firm conviction and expressed with confidence and certainty.

### AgreeableOpinions Class

- Represents viewpoints that are generally accepted or approved by most people in a conversation or discussion.

### ExtensivelyDiscussedOpinions Class

- Represents viewpoints or beliefs that have been thoroughly examined, debated, or talked about in a conversation or discussion.

### NegativeOpinions Class

- Represents viewpoints or beliefs held with a strong suspicion or doubt.

### Report Class

- Data model for a candidate report.
- Includes strong opinions, agreeable opinions, extensively discussed opinions, and negative opinions.
- Provides methods to generate a dictionary representation of the report and to pretty print the report.

## AdvisorReport Class

- Initializes with a list of agents and the number of conversation rounds.
- Generates advisor messages in a round-robin fashion and utilizes an OpenAI model and program to analyze the messages.
- Utilizes the SentenceTransformer model for sentence embeddings and PolyFuzz for opinion matching.
- Provides methods to generate and format the report.

## Usage

1. **Initialization**: Create an instance of the `AdvisorReport` class, providing a list of agents and the number of conversation rounds.

2. **Message Generation**: Use the `get_advisor_messages` method to generate advisor messages in a round-robin fashion.

3. **Analysis**: Utilize the OpenAI model and program to analyze the messages, generating a report.

4. **Sentence Embeddings and Matching**: The `SentenceTransformer` model is used for sentence embeddings, and PolyFuzz is employed for opinion matching.

5. **Report Generation**: The report is generated, including strong opinions, agreeable opinions, extensively discussed opinions, and negative opinions.

6. **Data Output**: The `generate` method outputs a Pandas DataFrame with matched opinions, source messages, and corresponding advisors.

## Note

- Ensure that the required libraries, including `polyfuzz`, `sentence_transformers`, and other dependencies, are installed before running the script.

- Adjust parameters such as the OpenAI model name, embedding model, and matching criteria based on specific requirements.

- The script is designed for analyzing opinions in a conversation, and the `AdvisorReport` class can be extended or modified for different use cases.