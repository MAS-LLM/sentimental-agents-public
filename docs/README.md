# Documentation

## Documentation Overview

Welcome to the documentation for the `sentimental-agents` project. This documentation provides an overview of the project structure, key modules, and instructions on using and understanding the simulation, sentiment analysis, and related utilities.

## Documentation Structure

The documentation is organized into several files, each corresponding to a specific module or aspect of the project. Here's a brief overview of the available documents:

### [Dialog Simulation](files/dialog.md)

This document provides information about the dialog simulation aspect of the project. It explains how the dialogue agents are initialized, the simulation loop, and how the conversation unfolds.

### [Main Simulation Execution](files/main.md)

Learn about the main script, `main.py`, which orchestrates the entire simulation process. Understand the various functions and their roles in simulating candidate conversations.

### [Non-Bayesian Sentiment Analysis](files/nonbayesian.md)

Explore the non-Bayesian sentiment analysis performed on agent messages during the simulation. This document explains how sentiment estimates are updated and tracked for each agent.

### [Opinion Analyzer](files/opinion_analyser.md)

Understand the opinion analysis module, which categorizes opinions into different types. Learn how opinions are extracted, grouped, and presented in the generated reports.

### [Sentiment Analysis](files/sentiment.md)

This document covers the sentiment analysis module, focusing on the `SentimentAnalyzer` class. Learn how sentiment classification is performed on individual opinions and how overall sentiment scores are calculated.

### [Simulation Utilities](files/simulation_utilities.md)

Explore the utility functions used throughout the simulation. Understand how agent information, system messages, and other components are generated, and how the simulation output is processed.

### [Templates and Data models](files/advisorybrief.md)

The `advisory_brief.py` file contains templates and data models for creating advisory briefs in the context of simulating dialogues involving various advisors.

## Input Examples

Check the [input_examples](input_examples/) directory for sample JSON files that demonstrate the expected input format for various simulation parameters. These examples can be used as a reference when creating input data for candidate simulations.

## Data

The [data](data/) directory contains data-related utilities, including a downloader, processor, and resume samples. The [data_utilities](data/data_utilities/) subdirectory further organizes input and output data, with a [Readme.md](data/data_utilities/Readme.md) providing additional information.

## Running the Simulation

To run the simulation, refer to the [main.py](main.py) script. Ensure that the required dependencies are installed by checking the [requirements.txt](requirements.txt) file. The simulation setup can be configured through the `.env` file.

## Customization

For further assistance or customization inquiries, refer to the [README.md](README.md) file in the project root.

---

**Note**: Please make sure to follow the documentation and adapt the project according to your specific use case. The structure provided aims to facilitate understanding and customization for diverse applications.