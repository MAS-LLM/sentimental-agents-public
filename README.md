# Sentimental Agents

This project implements a simulation and evaluation pipeline for **sentiment-aware agents** that deliberate, update opinions, and make collective decisions. It includes tools for dialog generation, sentiment analysis, evaluation, and visualization.


## Installation

We recommend using a **Python 3.10+ conda environment**.

```bash
conda create -n sentimental_agents python=3.10
conda activate sentimental_agents
pip install -r requirements.txt
```

##  Running the Project

### Single Run

`main.py` accepts several arguments to control experiments. At minimum, you must provide:

- `--data_path` : path to the input CSV file with job descriptions / resumes  
- `--output_dir` : directory where results will be stored  
- `--num_candidates` : number of candidates to simulate  
- `--num_rounds` : number of dialog rounds  
- `--temperature` : sampling temperature for LLMs  

Example:

```bash
python main.py --data_path data/input/sample.csv --output_dir output_files --num_candidates 1 --num_rounds 5 --temperature 0.7
```

### Batch Runs

You can launch multiple runs with default configs using:

```bash
./run.sh
```



