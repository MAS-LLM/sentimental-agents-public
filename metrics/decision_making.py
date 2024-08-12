import os
import json
import pandas as pd
import numpy as np

class DecisionMaker:
    """
    A class to make decisions based on sentiment analysis of agents' messages.
    """
    def __init__(self, directory: str, setup_file: str, score_type: str = "average"):
        """
        Initialize the DecisionMaker with a directory, setup file, and score type.

        Args:
            directory (str): The directory containing the simulation data.
            setup_file (str): Path to the simulation setup data JSON file.
            score_type (str, optional): The type of score to use. Defaults to "average".
        """
        self.directory = directory
        self.setup_file = setup_file
        self.score_type = score_type
        self.calculate_decision_metrics()

    def load_simulation_data(self):
        folders = [f for f in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, f))]

        # Sort the folders based on modification time
        sorted_folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(self.directory, f)))
        self.sim_data = dict()
        for folder in sorted_folders:
            with open(f"{self.directory}/{folder}/simulation_data.json", "r", encoding="utf-8") as jfile:
                data = json.load(jfile)
            self.sim_data[folder.split("_")[-1]] = data  

    def assign_scores(self):
        with open(self.setup_file, "r", encoding="utf-8") as jfile:
            setup_data = json.load(jfile)
        self.agents = setup_data["technical_advisors"]
        #self.agents = [x["name"] for x in list(self.sim_data.values())[0]["agent_data"]]
        self.score_data = []
        for key, candidate_data in self.sim_data.items():
            agent_scores = dict()
            for entry in candidate_data["agent_data"]:
                agent = entry["name"]
                if self.score_type == "average":
                    try:
                        sentiment_scores = [x['sentiment_data']['overall_sentiment'] for x in entry["messages"]]
                        score = sum(sentiment_scores) / len(sentiment_scores)
                    except ZeroDivisionError:
                        score = 0
                elif self.score_type == "last_value":
                    try:
                        score = entry["messages"][-1]['sentiment_data']['overall_sentiment']
                    except IndexError:
                        score = np.nan
                agent_scores[agent] = score
            agent_scores.update(
                {
                    "candidate_name": key
                    }
            )
            self.score_data.append(
                agent_scores
            )

        self.sdf = pd.DataFrame(self.score_data)
    def rank_list(self, intuition = False):
        """
        Rank agents based on their sentiment score.
        """
        if intuition:
            for agent in self.agents:
                self.sdf[f'{agent}_intuition_rank'] = self.sdf[f'{agent}_intuition'].rank(ascending=False)
        else:
            for agent in self.agents:
                self.sdf[f'{agent}_rank'] = self.sdf[agent].rank(ascending=False)
    
    def assign_valence(self) -> int:
        """
        Assign a valence based on the sentiment score.
        """
        for agent in self.agents:
            agent_valence = []
            try:
                for score in self.sdf[agent]:
                    if score < -0.5:
                        agent_valence.append(-1)
                    elif -0.5 <= score <= 0.5:
                        agent_valence.append(0)
                    else:
                        agent_valence.append(1)
                self.sdf[f"{agent}_valence"] = agent_valence
            except KeyError:
                self.sdf[f"{agent}_valence"] = np.nan

    def borda_count(self, intuition = False):
        """
        Assign points to agents based on their rank.
        """
        if intuition:
            rank_sums = [self.sdf[[f'{y}_intuition_rank' for y in self.agents]].loc[x].sum() for x in range(len(self.sdf))]
            sorted_lst = sorted(rank_sums, reverse=True)
            ranked_lst = [sorted_lst.index(x) + 1 for x in rank_sums]
            self.sdf["intuitive_rank"] = ranked_lst
        else:
            rank_sums = [self.sdf[[f'{y}_rank' for y in self.agents]].loc[x].sum() for x in range(len(self.sdf))]
            sorted_lst = sorted(rank_sums, reverse=True)
            ranked_lst = [sorted_lst.index(x) + 1 for x in rank_sums]
            self.sdf["borda_count"] = ranked_lst


    def get_tiered_list(self):
        """
        Assign a tier to agents based on their valence.
        """
        for agent in self.agents:
            agent_tier = []
            for valence in self.sdf[f"{agent}_valence"]:
                if valence == 1:
                    agent_tier.append(1)
                elif valence == 0:
                    agent_tier.append(2)
                else:
                    agent_tier.append(3)
            self.sdf[f"{agent}_tier"] = agent_tier

    def calculate_confidence(self):
        """
        Calculate the standard deviation of the sentiment scores.
        """
        std_list = []
        for key, candidate_data in self.sim_data.items():
            agent_confidence = dict()
            for entry in candidate_data["agent_data"]:
                agent = entry["name"]                
                sentiment_scores = [x['sentiment_data']['overall_sentiment'] for x in entry["messages"]]
                std = np.std(sentiment_scores)
                agent_confidence[f"{agent}_confidence"] = std
            std_list.append(agent_confidence)
        std_df = pd.DataFrame(std_list)
        self.sdf = pd.concat([self.sdf, std_df], axis=1)
            

    def intuitive_list(self):
        """
        Rank agents based on their intuition score.
        """
        for agent in self.agents:
            self.sdf[f'{agent}_intuition'] = self.sdf[agent] * self.sdf[f'{agent}_confidence']

        self.rank_list(intuition=True)
        self.borda_count(intuition=True)

    def save_results(self, filename: str = "decision_metrics.csv"):
        """
        Save the sentiment data frame (sdf) to a CSV file in the specified directory.

        Args:
            filename (str, optional): Name of the CSV file. Defaults to "decision_metrics.csv".
        """
        output_path = os.path.join(self.directory, filename)
        self.sdf.to_csv(output_path, index=False)
        print(f"Decision metrics saved to: {output_path}")


    def calculate_decision_metrics(self):
        self.load_simulation_data()
        self.assign_scores()
        self.assign_valence()
        self.rank_list()
        self.borda_count()
        self.get_tiered_list()
        self.calculate_confidence()
        self.intuitive_list()

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DecisionMaker on a specified folder")
    parser.add_argument("folder", type=str, help="Path to the folder containing simulation data")
    parser.add_argument("--setup_file", type=str, default="data/input/simulation_setup_data.json",
                        help="Path to the simulation setup data JSON file")
    parser.add_argument("--score_type", type=str, default="average", choices=["average", "last_value"],
                        help="Type of score to use (default: average)")
    parser.add_argument("--output_file", type=str, default="decision_metrics.csv",
                        help="Name of the output CSV file (default: decision_metrics.csv)")

    args = parser.parse_args()

    decision_maker = DecisionMaker(directory=args.folder, setup_file=args.setup_file, score_type=args.score_type)
    
    # Save the results to a CSV file
    decision_maker.save_results(args.output_file)
    
    # Print summary information
    print(f"Decision metrics calculated for data in: {args.folder}")
    print(f"Using setup file: {args.setup_file}")
    print(f"Using score type: {args.score_type}")
    print(f"Results saved to: {os.path.join(args.folder, args.output_file)}")