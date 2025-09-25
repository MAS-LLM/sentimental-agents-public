# Sentimental Agents

This project implements a simulation and evaluation pipeline for **sentimental agents** that deliberate, update opinions, and make collective decisions across various evaluation scenarios. It includes tools for dialog generation, sentiment analysis, evaluation, and visualization using multiple language models through Ollama.

## Prerequisites

### Ollama Setup (Required)
This project uses **Ollama** for local language model inference. **Ollama must be running** before executing any simulations.

1. **Install Ollama**: Follow instructions at https://ollama.ai
2. **Start Ollama service**:
   ```bash
   ollama serve
   ```
3. **Pull required models** (choose based on your hardware):
   ```bash
   # Lightweight models (good for testing)
   ollama pull deepseek-r1:1.5b
   ollama pull llama3.2:1b
   
   # Medium models (balanced performance)
   ollama pull llama3.1:8b
   ollama pull mistral:7b
   
   # Larger models (better quality, requires more RAM)
   ollama pull gpt-oss
   ollama pull gemma3:27b
   ```

### Python Environment
We recommend using **Python 3.10+ conda environment**.
```bash
conda create -n sentimental_agents python=3.10
conda activate sentimental_agents
pip install -r requirements.txt
```

## How the Simulation Works

The system simulates **multi-agent evaluation scenarios** where AI agents with different expertise collectively assess items for decision-making:

1. **Agent Generation**: Creates evaluators with distinct roles and specialized criteria
2. **Item Evaluation**: Agents discuss merits and drawbacks using their domain expertise  
3. **Sentiment Tracking**: Monitors emotional dynamics and opinion changes during deliberation
4. **Multi-Modal Analysis**: Tests different sentiment feedback modes:
   - `none`: No sentiment awareness
   - `own_sentiment`: Agents aware of their own emotional state
   - `others_sentiment`: Agents aware of others' emotional states

Each simulation runs across:
- **Multiple models**: Tests different LLMs for varied reasoning styles
- **Multiple temperatures**: Controls response randomness (0.0, 0.3, 0.7)
- **Multiple seeds**: Ensures statistical significance (30 random seeds)
- **Multiple feedback modes**: Compares sentiment-aware vs standard evaluation

## Supported Evaluation Domains

The framework supports various evaluation contexts through configurable templates:

### 1. Job Candidate Review
- **Evaluators**: E.g CFO, VP of Engineering, Department Manager
- **Input**: Candidate resume and job description
- **Output**: Hiring recommendation with pros/cons analysis

### 2. Academic Paper Review  
- **Evaluators**: E.g Senior Researcher, Associate Editor, Domain Expert
- **Input**: Paper abstract/content and venue standards
- **Output**: Publication recommendation with technical assessment

*Additional domains can be configured by modifying the prompt templates and simulation setup.*

## Data Structure

### Input Data
```
data/input/
├── simulation_setup_data.json       # Domain configuration (job/venue/publication)
└── default_sample_1.csv          # Item data (candidates/papers/articles)
```

### Output Data
```
output_files/
└── YYYYMMDD_HHMMSS_model_full_experiment/
    └── Item_001/                    # Individual item results
        ├── none_temp0.0_seed10/     # Condition-specific results
        │   └── simulation_data.json
        ├── own_sentiment_temp0.3_seed20/
        └── others_sentiment_temp0.7_seed30/
```

## Usage

### Single Run (Testing)
For a quick test with one item and one model:

```bash
python main.py \
  --simulation_setup_data data/input/simulation_setup_data.json \
  --candidate_csv data/input/candidate_sample_1.csv \
  --num_processes 4
```

### Full Experiment (Production)
For comprehensive multi-model experiments:

```bash
./run.sh
```

**Note**: Each experiment covers **one model at a time** but tests all combinations of temperatures, seeds, and feedback modes. The script processes papers sequentially to avoid overwhelming system resources.

### Batch Processing
The `run.sh` script automatically:
- Detects your platform (Linux/macOS/Windows)
- Sets optimal CPU affinity (Linux only)
- Processes all CSV files in `data/input/`
- Retries failed runs up to 3 times
- Runs comprehensive evaluation after simulation

## Configuration

### Models Available
Current supported models (configured in `utilities.py`):
- `deepseek-r1:1.5b` - Fast, lightweight
- `llama3.2:1b` - Ultra-lightweight
- `llama3.1:8b` - Balanced performance
- `mistral:7b` - Strong reasoning
- `gpt-oss` - Open-source GPT variant
- `gemma3:27b` - High-quality, resource-intensive

### Experiment Parameters
- **Seeds**: 30 random seeds (10-300) for statistical significance
- **Temperatures**: 0.0 (deterministic), 0.3 (balanced), 0.7 (creative)
- **Feedback modes**: 3 sentiment awareness conditions
- **Max rounds**: 10 discussion rounds per simulation

## Evaluation and Analysis

After simulations complete, the system automatically generates:
- **Confidence intervals** (95% CI) for all metrics
- **Statistical significance tests** between conditions
- **Sentiment evolution plots** showing opinion dynamics
- **Agent-level analysis** of individual reviewer behavior
- **Cross-condition comparisons** of feedback mode effectiveness

Results include both raw per-seed data and aggregated statistics for publication-ready analysis.

## Troubleshooting

### Common Issues
- **"Connection refused"**: Ensure Ollama is running (`ollama serve`)
- **Model not found**: Pull the required model (`ollama pull model-name`)
- **Memory issues**: Use smaller models or reduce batch size
- **Slow performance**: Check CPU affinity settings in `run.sh`

### Monitoring
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Monitor system resources
htop  # Linux
top   # macOS/Linux
```

The system provides detailed logging and progress indicators throughout execution.
