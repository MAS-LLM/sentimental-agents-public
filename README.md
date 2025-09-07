# Sentimental Agents: Setup, Simulation and Evaluation Guide

This project implements a multi-agent dialogue simulation system for candidate evaluation using Python 3.10.12 and conda for environment management.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Git (to clone the repository)

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create a new conda environment with Python 3.10.12
conda create -n simulation-env python=3.10.12 -y

# Activate the environment
conda activate simulation-env
```

### 2. Install Dependencies

```bash
# Install packages from requirements.txt
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Add your configuration to `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 4. Prepare Data Files

Ensure your data files are in the correct location:
- Place `simulation_setup_data.json` in `data/input/`
- Place your CSV files (e.g., `default_sample_1.csv`) in `data/input/`

## How the Simulation Works

### Overview
The simulation runs a two-stage evaluation process for candidate assessment:

1. **Single LLM Evaluation**: A single AI model evaluates each candidate
2. **Multi-Agent Simulation**: Multiple AI agents with different perspectives discuss each candidate

### Stage 1: Single LLM Evaluation
- Uses a unified advisor approach with one AI model (default: GPT-4o-mini)
- Evaluates candidates based on predefined criteria, priorities, and job requirements
- Generates a single opinion and sentiment score for each candidate
- Outputs results to `single_llm_evaluation_results.csv`

### Stage 2: Multi-Agent Simulation
- Creates multiple AI agents with distinct roles (technical advisors)
- Each agent has unique:
  - **Descriptions**: Role-specific background and expertise
  - **Priorities**: What each agent values most in candidates
  - **Criteria**: Specific evaluation standards for the role
- Agents engage in dialogue rounds discussing each candidate
- **Sentiment tracking**: Monitors how each agent's opinion evolves during discussion
- **Stopping conditions**: Simulation ends when consensus is reached or maximum rounds completed

### Key Components
- **DialogueSimulator**: Manages conversation flow between agents
- **SentimentAgent**: Tracks emotional and opinion changes throughout discussion
- **Configuration**: Adjustable parameters for temperature, rounds, and behavior

## Evaluation Metrics

The system generates comprehensive metrics to analyze simulation results:

### 1. **Sentiment Analysis**
- **Definition**: Tracks how agent opinions (positive/negative) change during discussions
- **Single vs Multi-Agent Comparison**: Compares sentiment between single LLM and multi-agent approaches
- **Variance**: Measures disagreement levels among agents

### 2. **Emergent Behavior Analysis**
- **Polarization**: Sentiment variance across agents (higher = more disagreement)
- **Consensus**: Agent synchronization scores (higher = more agreement)
- **Groupthink Detection**: Identifies when agents converge too quickly

### 3. **Prolificness Score**
- **Definition**: Number of unique arguments each agent type generates
- **Purpose**: Measures how much each role contributes to discussions

### 4. **Nuance Score**
- **Definition**: Topic diversity analysis using Latent Dirichlet Allocation (LDA)
- **Purpose**: Identifies different themes and topics discussed per candidate
- **Output**: Excel file with top words per topic for each candidate

### 5. **Similarity Analysis**
- **Intra-Agent Similarity**: How consistent each agent's messages are with themselves
- **Inter-Agent Similarity**: How similar different agents' messages are to each other
- **Purpose**: Detects redundancy or unique perspectives

### 6. **Drift Analysis**
- **Definition**: How much agent messages deviate from their original system prompts
- **Purpose**: Measures if agents stay true to their assigned roles

### 7. **Defensibility Check**
- **Definition**: How well agent arguments are supported by candidate resume data
- **Method**: Uses semantic similarity between arguments and resume content
- **Output**: Scores indicating argument-evidence alignment

### 8. **Cognitive Bias Metrics**
- **Bias Gap**: Difference between single LLM and multi-agent sentiment
- **Inconsistency Index**: Combines bias gap and variance for overall inconsistency measure
- **Extremity Analysis**: Compares absolute sentiment values between approaches

### 9. **Statistical Validation**
- **Bland-Altman Analysis**: Agreement between single and multi-agent sentiment
- **Cosine Similarity**: Vector similarity between sentiment approaches

## Running the Simulation

### Single Run
```bash
# Activate environment
conda activate simulation-env

# Run simulation with CSV file
python main.py \
    --simulation_setup_data data/input/simulation_setup_data.json \
    --candidate_csv data/input/default_sample_1.csv \
    --num_processes 4

# Run simulation with single candidate
python main.py \
    --simulation_setup_data data/input/simulation_setup_data.json \
    --candidate_name "John Doe" \
    --candidate_bio "Software engineer with 5 years experience..."
```

### Batch Runs
```bash
# Make run script executable
chmod +x run.sh

# Run batch simulations
./run.sh
```

## Configuration

### Process and CPU Settings

Edit the configuration variables at the top of `run.sh`:

```bash
export NUM_PROCS=4              # Number of parallel worker processes
MULTIAGENT_CORES="8-15"         # CPU cores to use (Linux only)
MAX_ATTEMPTS=3                  # Retry attempts for failed runs
```

**How to determine optimal settings for your machine:**

1. **NUM_PROCS** (Number of processes):
   - **Check your CPU cores:** `nproc` (Linux), `sysctl -n hw.ncpu` (macOS), or Task Manager (Windows)
   - **Recommended:** Use 50-75% of your CPU cores
   - **Examples:**
     - 4-core machine: `NUM_PROCS=2` or `NUM_PROCS=3`
     - 8-core machine: `NUM_PROCS=4` or `NUM_PROCS=6`
     - 16-core machine: `NUM_PROCS=8` or `NUM_PROCS=12`

2. **MULTIAGENT_CORES** (CPU affinity - Linux only):
   - **Format:** "start-end" or "core1,core2,core3"
   - **Example for 16-core machine:** Reserve cores 0-7 for system, use 8-15 for simulation
   - **Not available on macOS/Windows** - script will automatically skip this feature

3. **Memory considerations:**
   - Each process uses additional RAM
   - Monitor memory usage: `htop` (Linux), Activity Monitor (macOS), Task Manager (Windows)
   - Reduce `NUM_PROCS` if you run out of memory

### Platform-Specific Notes

#### Linux
- Full feature support including CPU affinity control
- Use `htop` or `top` to monitor resource usage

#### macOS
- CPU affinity (`taskset`) not available - feature automatically disabled
- Use Activity Monitor to check resource usage

#### Windows
- Run script in Git Bash, WSL, or similar bash environment
- CPU affinity not supported via bash - feature automatically disabled
- Use Task Manager to monitor resource usage

## Output Files

Results are saved in timestamped directories under `output_files/`:

```
output_files/
└── YYYYMMDD_HHMMSS/           # Timestamp of simulation run
    ├── candidate_name/         # Individual candidate results
    │   └── simulation_data.json
    ├── sentiment_comparison_single_vs_multiagent.csv
    ├── emergent_behavior_metrics.csv
    ├── cognitive_bias_metrics.csv
    ├── nuance_scores.xlsx
    ├── defensibility_scores.xlsx
    ├── drift_df.csv
    ├── bland_altman_stats.csv
    ├── cosine_similarity.txt
```

### Key Output Files:
- **`simulation_data.json`**: Complete conversation history, sentiment data, and costs per candidate
- **`sentiment_comparison_*.csv`**: Comparison between single LLM and multi-agent sentiment scores
- **`emergent_behavior_metrics.csv`**: Polarization and consensus metrics
- **`cognitive_bias_metrics.csv`**: Bias analysis between evaluation approaches
- **`nuance_scores.xlsx`**: Topic analysis with LDA results per candidate
- **`defensibility_scores.xlsx`**: Argument-evidence alignment scores
- **Various PNG plots**: Visual representations of all metrics


## Troubleshooting

### Common Issues

1. **Environment activation fails:**
   ```bash
   # Ensure conda is in PATH
   conda --version
   
   # If not found, add to PATH or restart terminal
   ```

2. **Package installation errors:**
   ```bash
   # Update pip
   pip install --upgrade pip
   
   # Install packages individually if needed
   pip install package_name
   ```

3. **Script permission errors (Linux/macOS):**
   ```bash
   chmod +x run.sh
   ```

4. **Python command not found:**
   ```bash
   # Try different Python commands
   python --version    # or
   python3 --version
   ```

### Performance Optimization

- **Start with lower `NUM_PROCS`** and gradually increase
- **Monitor system resources** during execution
- **Adjust based on your hardware** and available memory
- **Consider running overnight** for large batch jobs

## Environment Management

```bash
# List environments
conda env list

# Activate environment
conda activate simulation-env

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n simulation-env
```