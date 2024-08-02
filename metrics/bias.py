import json
import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def load_simulation_data(directory: str = "output_files") -> dict:
    """
    Load simulation data from a directory.

    Args:
    directory (str): Directory path where the simulation data is stored. Default is "output_files".

    Returns:
    dict: A dictionary containing the simulation data.
    """
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Sort the folders based on modification time
    sorted_folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    sim_data = dict()
    for folder in sorted_folders:
        with open(f"{directory}/{folder}/simulation_data.json", "r", encoding="utf-8") as jfile:
            data = json.load(jfile)
        sim_data[folder.split("_")[-1]] = data
    return sim_data

def get_data(candidate_data: dict) -> pd.DataFrame:
    """
    Get data from candidate data.

    Args:
    candidate_data (dict): Dictionary containing the candidate data.

    Returns:
    pd.DataFrame: A DataFrame containing the data.
    """
    data = candidate_data['non_bayesian_data']
    agents = list(data["sentiment_data"].keys())
    rounds = len(data["sentiment_data"][agents[0]])

    sentiment = []
    input_agent = []
    for i in range(1, rounds):
        for agent in agents:
            for k in range(2):
                sentiment.append(data["sentiment_data"][agent][i])
                input_agent.append(agent)

    change = []
    output_agent = []
    rounds_tracker = []
    for i in range(1, rounds):    
        for j, agent in enumerate(agents):
            if i > 1:
                for k in range(2):
                        change.append(data["change"][agent][i])
                        output_agent.append(agent)
                        rounds_tracker.append(i)
            elif (i == 1) and (j == 0):
                change.append(data["change"][agents[1]][1])
                output_agent.append(agents[1])
                rounds_tracker.append(i)
            elif (i == 1) and (j in [1,2]):
                change.append(data["change"][agents[2]][1])
                output_agent.append(agents[2])
                rounds_tracker.append(i)
    candidate = [candidate_data["summarized_output"]["Candidate Name"] for _ in sentiment]

    df = pd.DataFrame(
        data = {
            "sentiment": sentiment,
            "change": change + [np.nan for x in range(len(sentiment) - len(change))],
            "candidate" : candidate,
            "input_agent" : input_agent,
            "output_gent" : output_agent + [np.nan for x in range(len(sentiment) - len(change))],
            "round" : rounds_tracker + [np.nan for x in range(len(sentiment) - len(change))],
        }
    )
    # drop all rows with nan values
    df.dropna(inplace=True)
    return df

def generate_bias_data(directory, how: str = "all", index: int = 0) -> dict:
    """
    Generate bias data.

    Args:
    directory (str): Directory path where the simulation data is stored.
    how (str): How to generate the bias data. Options are "single" or "all". Default is "all".
    index (int): Index of the candidate data to use. Default is 0.

    Returns:
    dict: A dictionary containing the bias data for each candidate.
    """
    sim_data = load_simulation_data(directory=directory)
    if how == "single":
        candidate_data = list(sim_data.values())[index]
        return {list(sim_data.keys())[index]: get_data(candidate_data)}
    elif how == "all":
        return {candidate: get_data(data) for candidate, data in sim_data.items()}

def main():
    parser = argparse.ArgumentParser(description="Generate bias data from simulation output.")
    parser.add_argument("directory", help="Directory containing simulation output files")
    parser.add_argument("--how", choices=["single", "all"], default="all", help="How to generate the bias data")
    parser.add_argument("--index", type=int, default=0, help="Index of the candidate data to use (only for 'single' mode)")
    args = parser.parse_args()

    bias_data = generate_bias_data(args.directory, how=args.how, index=args.index)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.how == "all":
        # Generate Excel file with a sheet for each candidate
        output_filename = f"bias_data_all_{timestamp}.xlsx"
        output_path = os.path.join(args.directory, output_filename)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for candidate, df in bias_data.items():
                df.to_excel(writer, sheet_name=candidate, index=False)
        
        print(f"Bias data for all candidates saved to {output_path}")
    
    elif args.how == "single":
        # Generate CSV file with candidate name in filename
        candidate = list(bias_data.keys())[0]
        df = bias_data[candidate]
        
        output_filename = f"bias_data_{candidate}_{timestamp}.csv"
        output_path = os.path.join(args.directory, output_filename)
        
        df.to_csv(output_path, index=False)
        print(f"Bias data for {candidate} saved to {output_path}")

if __name__ == "__main__":
    main()