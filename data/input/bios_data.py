import pandas as pd
from datasets import load_dataset

# 1) Load & filter
ds = load_dataset("Elfsong/Bias_in_Bios")
filtered = ds['train'].filter(lambda x: x['profession'] == 'software_engineer')
print(len(filtered), "records for software engineers")

# 2) Pull out the fields we care about
records = [
    {'job_title': item['profession'], 'resume': item['hard_text']}
    for item in filtered
]
df = pd.DataFrame(records)

# 3) Generate 30 different random samples of size 10
sample_size = 10
num_samples = 1

for i in range(6, num_samples + 6):
    # draw without a fixed seed so each sample is different
    df_sample = df.sample(n=sample_size).reset_index(drop=True)

    # compute base offset: (file_index-1)*10
    base = (i - 1) * sample_size

    # assign Cand_001…Cand_300
    df_sample['candidate_name'] = df_sample.index.to_series().apply(
        lambda idx: f"Cand_{base + idx + 1:03d}"
    )

    # reorder and save
    df_out = df_sample[['candidate_name', 'job_title', 'resume']]
    out_path = f"sampl_{i}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  (names {df_out['candidate_name'].iloc[0]}–{df_out['candidate_name'].iloc[-1]})")
