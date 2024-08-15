import json
import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for better box plots
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.cluster import KMeans
def load_simulation_data(directory: str = "output_files") -> dict:
    """
    Load simulation data from a directory.
    """
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
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
            elif (i == 1) and (j in [1, 2]):
                change.append(data["change"][agents[2]][1])
                output_agent.append(agents[2])
                rounds_tracker.append(i)
    candidate = [candidate_data["summarized_output"]["Candidate Name"] for _ in sentiment]

    df = pd.DataFrame(
        data={
            "sentiment": sentiment,
            "change": change + [np.nan for x in range(len(sentiment) - len(change))],
            "candidate": candidate,
            "input_agent": input_agent,
            "output_gent": output_agent + [np.nan for x in range(len(sentiment) - len(change))],
            "round": rounds_tracker + [np.nan for x in range(len(sentiment) - len(change))],
        }
    )
    # drop all rows with nan values
    df.dropna(inplace=True)
    return df


def perform_regression(df: pd.DataFrame, sentiment_type: str):
    """
    Perform regression analysis for a given sentiment type.
    """
    if sentiment_type == 'positive':
        data = df[df['sentiment'] > 0]
    elif sentiment_type == 'negative':
        data = df[df['sentiment'] < 0]
    elif sentiment_type == 'saliency':
        threshold = df['change'].mean() + 2 * df['change'].std()
        data = df[abs(df['change']) > threshold]
    else:
        raise ValueError("Invalid sentiment type")

    X = data['sentiment'].values.reshape(-1, 1)
    y = data['change'].values

    # Fit linear regression model with intercept
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Get the slope, intercept, and R² value
    slope = model.coef_[0]
    intercept = model.intercept_  # Get the intercept
    r_squared = model.score(X, y)

    return slope, intercept, r_squared, data



def plot_combined_regression(df: pd.DataFrame, save_path: str):
    """
    Plot and save combined regression for positivity and negativity biases.
    """
    # Positive bias regression
    slope_pos, intercept_pos, r_squared_pos, df_pos = perform_regression(df, 'positive')
    # Negative bias regression
    slope_neg, intercept_neg, r_squared_neg, df_neg = perform_regression(df, 'negative')

    # Plotting positive bias data
    plt.scatter(df_pos['sentiment'], df_pos['change'], label='Positive Bias Data', alpha=0.5, color='blue')
    plt.plot(df_pos['sentiment'], df_pos['sentiment'] * slope_pos + intercept_pos, color='red',
             label=f'Positive Fit: y={slope_pos:.4f}x + {intercept_pos:.4f}\nR²={r_squared_pos:.4f}')

    # Plotting negative bias data
    plt.scatter(df_neg['sentiment'], df_neg['change'], label='Negative Bias Data', alpha=0.5, color='orange')
    plt.plot(df_neg['sentiment'], df_neg['sentiment'] * slope_neg + intercept_neg, color='green',
             label=f'Negative Fit: y={slope_neg:.4f}x + {intercept_neg:.4f}\nR²={r_squared_neg:.4f}')

    plt.title('Regression Analysis - Positivity vs Negativity Bias')
    plt.xlabel('Sentiment')
    plt.ylabel('Change')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_saliency_bias(df: pd.DataFrame, save_path: str):
    """
    Plot and save the saliency bias regression analysis.
    """
    # Saliency bias regression
    slope_sal, intercept_sal, r_squared_sal, df_sal = perform_regression(df, 'saliency')

    # Plotting saliency bias data
    plt.scatter(df_sal['sentiment'], df_sal['change'], label='Saliency Bias Data', alpha=0.5, color='purple')
    plt.plot(df_sal['sentiment'], df_sal['sentiment'] * slope_sal + intercept_sal, color='red',
             label=f'Saliency Fit: y={slope_sal:.4f}x + {intercept_sal:.4f}\nR²={r_squared_sal:.4f}')

    plt.title('Regression Analysis - Saliency Bias')
    plt.xlabel('Sentiment')
    plt.ylabel('Change')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_box_plots(df: pd.DataFrame, save_path: str):
    """
    Plot and save box plots for positivity, negativity, and saliency biases.
    """
    # Categorize bias types
    df['bias_category'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative')

    # Add saliency category
    threshold = df['change'].mean() + 2 * df['change'].std()
    df['bias_category'] = np.where(abs(df['change']) > threshold, 'saliency', df['bias_category'])

    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='bias_category', y='change', data=df)
    plt.title('Box Plot of Sentiment Changes by Cognitive Bias Category')
    plt.xlabel('Cognitive Bias Category')
    plt.ylabel('Sentiment Change')
    plt.savefig(save_path)
    plt.close()


def generate_bias_data(directory, how: str = "all", index: int = 0) -> dict:
    """
    Generate bias data.
    """
    sim_data = load_simulation_data(directory=directory)
    if how == "single":
        candidate_data = list(sim_data.values())[index]
        return {list(sim_data.keys())[index]: get_data(candidate_data)}
    elif how == "all":
        return {candidate: get_data(data) for candidate, data in sim_data.items()}


def plot_comprehensive_cognitive_bias(df: pd.DataFrame, save_path: str):
    """
    Plot cognitive biases using a density plot with indicators for positive, negative, and saliency bias.
    """
    # Create the plot with increased width
    plt.figure(figsize=(14, 10))  # Increase the width to 14 for more spacing on the sides

    # Density plot
    sns.kdeplot(data=df[df['sentiment'] >= 0], x='sentiment', y='change', cmap="Blues", shade=True,
                label='Positive Sentiment',
                clip=((0.01, df['sentiment'].max()), (df['change'].min(), df['change'].max())))
    sns.kdeplot(data=df[df['sentiment'] < 0], x='sentiment', y='change', cmap="Reds", shade=True,
                label='Negative Sentiment',
                clip=((df['sentiment'].min(), -0.01), (df['change'].min(), df['change'].max())))

    # Calculate bias indicators
    mean_sentiment = df['sentiment'].mean()
    mean_change = df['change'].mean()
    saliency_threshold = df['change'].std()

    # Calculate summary statistics
    positive_bias = (df['sentiment'] > 0).mean()
    negative_bias = (df['sentiment'] < 0).mean()
    saliency_bias = (abs(df['change']) > saliency_threshold).mean()

    # Add bias indicator lines
    plt.axvline(x=mean_sentiment, color='green', linestyle='--', label='Mean Sentiment')
    plt.axhline(y=mean_change, color='purple', linestyle='--', label='Mean Change')
    plt.axhline(y=saliency_threshold, color='orange', linestyle='--', label='Saliency Threshold')
    plt.axhline(y=-saliency_threshold, color='orange', linestyle='--')

    # Set x-axis limits with some padding
    sentiment_min = df['sentiment'].min()
    sentiment_max = df['sentiment'].max()
    plt.xlim(sentiment_min - 0.1 * abs(sentiment_min), sentiment_max + 0.1 * abs(sentiment_max))

    # Add annotations for bias
    plt.text(0.98, 0.98, f'Positive Bias: {positive_bias:.2f}', transform=plt.gca().transAxes, ha='right', va='top', fontweight='bold')
    plt.text(0.98, 0.93, f'Negative Bias: {negative_bias:.2f}', transform=plt.gca().transAxes, ha='right', va='top', fontweight='bold')
    plt.text(0.98, 0.88, f'Saliency Bias: {saliency_bias:.2f}', transform=plt.gca().transAxes, ha='right', va='top', fontweight='bold')

    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Change', fontsize=14)
    plt.title('Density Plot of Sentiment vs. Change with Bias Indicators', fontsize=18)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cognitive_bias_single_heatmap(df: pd.DataFrame, save_path: str):
    """
    Plot cognitive biases using a distribution-like heatmap for both positive and negative sentiment points.
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a heatmap using kernel density estimation with a diverging color palette
    sns.kdeplot(data=df, x='sentiment', y='change', cmap='coolwarm', shade=True, cbar=True, ax=ax)

    # Calculate bias indicators
    mean_sentiment = df['sentiment'].mean()
    mean_change = df['change'].mean()
    saliency_threshold = df['change'].std()

    # Add bias indicator lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Sentiment Boundary')
    ax.axvline(x=mean_sentiment, color='green', linestyle='--', label='Mean Sentiment')
    ax.axhline(y=mean_change, color='purple', linestyle='--', label='Mean Change')
    ax.axhline(y=saliency_threshold, color='orange', linestyle='--', label='Saliency Threshold')
    ax.axhline(y=-saliency_threshold, color='orange', linestyle='--')

    # Calculate bias metrics
    positive_bias = (df['sentiment'] > 0).mean()
    negative_bias = (df['sentiment'] < 0).mean()
    saliency_bias = (abs(df['change']) > saliency_threshold).mean()

    # Set titles and labels
    ax.set_title('Sentiment vs. Change Bias Heatmap')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Change')

    # Add annotations for bias
    bias_text = f"Positive Bias: {positive_bias:.2f}\n"
    bias_text += f"Negative Bias: {negative_bias:.2f}\n"
    bias_text += f"Saliency Bias: {saliency_bias:.2f}\n"
    bias_text += f"Mean Sentiment: {mean_sentiment:.2f}"
    ax.text(0.02, 0.98, bias_text, transform=ax.transAxes, ha='left', va='top', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.5))

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Calculate summary statistics
    summary = pd.DataFrame({
        'Metric': ['Positive Bias', 'Negative Bias', 'Saliency Bias',
                   'Mean Sentiment', 'Mean Change', 'Saliency Threshold'],
        'Value': [positive_bias, negative_bias, saliency_bias,
                  mean_sentiment, mean_change, saliency_threshold]
    })

    # Save summary to CSV
    summary_path = save_path.rsplit('.', 1)[0] + '_summary.csv'
    summary.to_csv(summary_path, index=False)

    print(f"Cognitive bias heatmap saved to {save_path}")
    print(f"Summary statistics saved to {summary_path}")
    print("\nBias Metrics:")
    print(f"Positive Bias: {positive_bias:.3f}")
    print(f"Negative Bias: {negative_bias:.3f}")
    print(f"Saliency Bias: {saliency_bias:.3f}")
    print(f"Mean Sentiment: {mean_sentiment:.3f}")
    print(f"Mean Change: {mean_change:.3f}")
    print(f"Saliency Threshold: {saliency_threshold:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Generate bias data from simulation output.")
    parser.add_argument("directory", help="Directory containing simulation output files")
    parser.add_argument("--how", choices=["single", "all"], default="all", help="How to generate the bias data")
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the candidate data to use (only for 'single' mode)")
    args = parser.parse_args()

    bias_data = generate_bias_data(args.directory, how=args.how, index=args.index)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.how == "all":
        # Generate Excel file with a sheet for each candidate
        output_filename = f"bias_data_all_{timestamp}.xlsx"
        output_path = os.path.join(args.directory, output_filename)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            combined_df = pd.concat(bias_data.values(), ignore_index=True)
            for candidate, df in bias_data.items():
                df.to_excel(writer, sheet_name=candidate, index=False)

        # # Save comprehensive cognitive bias visualization
        plot_comprehensive_cognitive_bias(combined_df,
                                          os.path.join(args.directory, f"cognitive_bias_comprehensive_{timestamp}.png"))
        # # Call the new heatmap function
        # heatmap_output_path = os.path.join(args.directory, f"cognitive_bias_heatmap_{timestamp}.png")
        # plot_cognitive_bias_single_heatmap(combined_df, heatmap_output_path)
        # # Save combined regression plots
        # print("plotting combined regression")
        # plot_combined_regression(combined_df, os.path.join(args.directory, f"combined_bias_{timestamp}.png"))
        # plot_saliency_bias(combined_df, os.path.join(args.directory, f"saliency_bias_{timestamp}.png"))
        #
        # # Save box plots for cognitive biases
        # plot_box_plots(combined_df, os.path.join(args.directory, f"box_plots_bias_{timestamp}.png"))

        print(f"Bias data and plots for all candidates saved to {output_path} and {args.directory}")

    elif args.how == "single":
        # Generate CSV file with candidate name in filename
        candidate = list(bias_data.keys())[0]
        df = bias_data[candidate]

        output_filename = f"bias_data_{candidate}_{timestamp}.csv"
        output_path = os.path.join(args.directory, output_filename)

        df.to_csv(output_path, index=False)

        # Save single candidate plots
        plot_combined_regression(df, os.path.join(args.directory, f"combined_bias_{candidate}_{timestamp}.png"))
        plot_saliency_bias(df, os.path.join(args.directory, f"saliency_bias_{candidate}_{timestamp}.png"))

        # Save box plots for cognitive biases
        plot_box_plots(df, os.path.join(args.directory, f"box_plots_bias_{candidate}_{timestamp}.png"))

        print(f"Bias data and plots for {candidate} saved to {output_path} and {args.directory}")

if __name__ == "__main__":
    main()

