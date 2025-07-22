# Sentimental Agents: Simulation and Evaluation

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
    ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create and '.env' file in the root directory of the project and add the following environment variables:
   ```bash
   OPENAI_API_KEY=""
   OPENAI_MODEL="gpt-4o-mini"
   ALPHAVANTAGE_API_KEY=""
   METAPHOR_API_KEY=""
    ```
## Running the Project

A convenience script `run.sh` is included to simplify running simulations and evaluations:

```bash
bash run.sh

   ```

[//]: # (6. Run the bias evaluation:)

[//]: # (   ```bash)

[//]: # (    python3 metrics/bias.py output_files/20240811_093714 --how all)

[//]: # (    ```)

[//]: # (8. Run the decision-making evaluation:)

[//]: # (```bash)

[//]: # (   python3 metrics/decision_making.py output_files/20240802_095722)

[//]: # ()
[//]: # (   ```)

---

## Dependencies and Environment Setup for LLMs

This project integrates multiple Large Language Models (LLMs) through different APIs and local models:

- **OpenAI GPT Models**  
  Requires the `openai` Python package and a valid OpenAI API key.

- **Anthropic Claude Models**  
  Requires the `anthropic` Python package and a valid Anthropic API key.

- **Local LLMs (e.g., LLaMA, others)**  
  Use `transformers`, `torch`, and optionally `sentencepiece` and `tokenizers` for local model inference.

## System Requirements

- **GPU:** Recommended for local LLMs to run efficiently (NVIDIA CUDA drivers + compatible PyTorch)

- **CPU:** Local LLMs can run on CPU but slower; llama.cpp or other optimized runtimes can help

## Environment Variables

Set API keys in `.env` file:


