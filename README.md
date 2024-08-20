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
4. Run the application:
```bash
   python3 main.py --simulation_setup_data "data/input/simulation_setup_data.json" --candidate_csv "data/input/10sample.csv"
   ```
5. Run the evaluation:
   ```bash
   python3 metrics/evaluation.py output_files/20240811_093714 data/input/samples.csv

   ```
6. Run the bias evaluation:
   ```bash
    python3 metrics/bias.py output_files/20240811_093714 --how all
    ```
8. Run the decision-making evaluation:
```bash
   python3 metrics/decision_making.py output_files/20240802_095722

   ```

