import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(10)

# Simulate convergence with the existing function
def simulate_convergence(alpha, tolerance, sample_new_evidence, max_rounds=20):
    prior = 0.0
    round_count = 0

    for _ in range(max_rounds):
        round_count += 1
        new_evidence = sample_new_evidence()
        updated_sentiment = alpha * new_evidence + (1 - alpha) * prior
        updated_sentiment = np.clip(updated_sentiment, -1, 1)
        change = abs(updated_sentiment - prior)
        if change < tolerance:
            break
        prior = updated_sentiment

    return round_count

alpha_values = np.arange(0.1, 1.0, 0.1)
tolerance_values = np.arange(0.001, 0.1, 0.01)

def sample_new_evidence():
    return np.random.normal(loc=0, scale=0.1)

median_convergence_data = np.zeros((len(alpha_values), len(tolerance_values)))
variance_convergence_data = np.zeros((len(alpha_values), len(tolerance_values)))

# Collect data for stability analysis
for i, alpha in enumerate(alpha_values):
    for j, tolerance in enumerate(tolerance_values):
        convergence_rounds = [simulate_convergence(alpha, tolerance, sample_new_evidence) for _ in range(100)]
        median_convergence_data[i, j] = np.median(convergence_rounds)
        variance_convergence_data[i, j] = np.var(convergence_rounds)

# Combine the data to identify stable regions
stability_map = (median_convergence_data < 5) & (variance_convergence_data < 2)

# Plot the median convergence rounds
plt.figure(figsize=(10, 8))
sns.heatmap(median_convergence_data,
            xticklabels=np.round(tolerance_values, 3),
            yticklabels=np.round(alpha_values[::-1], 2),
            annot=True, fmt=".0f", cmap="summer",
            cbar_kws={'label': 'Median Number of Rounds to Convergence'})
plt.xlabel("Tolerance")
plt.ylabel("Alpha")
plt.title("Median Convergence Rounds across Alpha and Tolerance Values")
plt.yticks(ticks=np.arange(len(alpha_values))+0.5, labels=np.round(alpha_values[::-1], 2))
plt.savefig("median_convergence.png")
plt.show()

# Plot the variance of convergence rounds
plt.figure(figsize=(10, 8))
sns.heatmap(variance_convergence_data,
            xticklabels=np.round(tolerance_values, 3),
            yticklabels=np.round(alpha_values[::-1], 2),
            annot=True, fmt=".2f", cmap="YlOrBr",
            cbar_kws={'label': 'Variance of Rounds to Convergence'})
plt.xlabel("Tolerance")
plt.ylabel("Alpha")
plt.title("Variance of Convergence Rounds across Alpha and Tolerance Values")
plt.yticks(ticks=np.arange(len(alpha_values))+0.5, labels=np.round(alpha_values[::-1], 2))
plt.savefig("variance_convergence.png")
plt.show()

# Plot the stability map
plt.figure(figsize=(10, 8))
sns.heatmap(stability_map.astype(int),
            xticklabels=np.round(tolerance_values, 3),
            yticklabels=np.round(alpha_values[::-1], 2),
            annot=True, fmt="d", cmap="coolwarm",
            cbar_kws={'label': 'Stability Region (1=Stable, 0=Unstable)'})
plt.xlabel("Tolerance")
plt.ylabel("Alpha")
plt.title("Stability Map across Alpha and Tolerance Values")
plt.yticks(ticks=np.arange(len(alpha_values))+0.5, labels=np.round(alpha_values[::-1], 2))
plt.savefig("stability_map.png")
plt.show()
