import pandas as pd
from datasets import load_dataset
import itertools

# Load the dataset
ds = load_dataset("Elfsong/Bias_in_Bios")

# Filter the dataset for 'software_engineer' profession
filtered_data = ds['train'].filter(lambda x: x['profession'] == 'software_engineer')


# Function to generate candidate names like "Candidate A", "Candidate B", etc.
def generate_candidate_names():
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for length in range(1, 5):  # Extend the range to allow for more names
        for name in itertools.product(letters, repeat=length):
            yield 'Candidate ' + ''.join(name)


# Generate candidate names
name_generator = generate_candidate_names()

# Extract relevant information and assign unique candidate names
data = []
for item in filtered_data:
    try:
        candidate_name = next(name_generator)
    except StopIteration:
        raise ValueError("Ran out of unique candidate names. Increase the length of the generated names.")

    job_title = item['profession']
    resume = item['hard_text']
    data.append({'candidate_name': candidate_name, 'job_title': job_title, 'resume': resume})

# Convert to DataFrame
df = pd.DataFrame(data)

# Shuffle and select a random sample of candidates
sample_size = 50  # Default sample size
df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

# Save the full dataset to CSV
# df.to_csv('software_engineers.csv', index=False)

# Save the random sample to a separate CSV
df_sample.to_csv('50sample.csv', index=False)

print(f"Filtered data saved to 'software_engineers.csv' and a random sample to '50sample.csv'")