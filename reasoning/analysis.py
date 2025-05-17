import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import numpy as np
import seaborn as sns

# Step 1: Load and combine datasets
base_path = "CohenQu/continue_vs_terminate_DAPO-Math-en"
all_datasets = []

for i in range(100):
    subset_name = f"DeepScaleR-1.5B-Preview_{i*20}_{i*20 + 20}"
    try:
        ds = load_dataset(base_path, name=subset_name, split="test")
        all_datasets.append(ds)
    except Exception:
        continue  # silently skip missing subsets

# Combine all datasets
if not all_datasets:
    raise ValueError("No datasets were successfully loaded.")

combined = concatenate_datasets(all_datasets)

# Step 2: Process progress and other fields
def parse_progress(example):
    try:
        numerator, denominator = map(int, example['progress'].split('/'))
        example['progress_ratio'] = numerator / denominator
    except:
        example['progress_ratio'] = None
    return example

combined = combined.map(parse_progress)
df = combined.to_pandas()
df = df.dropna(subset=["progress_ratio", "mean_reward", "action"])

# Step 3: Pivot to compute reward difference
pivot_df = df.pivot_table(index="progress_ratio", columns="action", values="mean_reward", aggfunc="mean").dropna()
pivot_df["reward_diff"] = pivot_df["Continue"] - pivot_df["Terminate"]
# Step 4: Compute ratios
gt_0 = (pivot_df["reward_diff"] > 0).mean()
lt_0 = (pivot_df["reward_diff"] < 0).mean()
eq_0 = (pivot_df["reward_diff"] == 0).mean()
print(f"gt_0: {gt_0}, lt_0: {lt_0}, and eq_0: {eq_0}")
pivot_df = pivot_df.reset_index()



# Step 4: Scatter plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pivot_df, x="progress_ratio", y="reward_diff", alpha=0.6)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Progress (int_1 / int_2)")
plt.ylabel("Reward Difference (Continue - Terminate)")
plt.title("Reward Difference vs. Progress")
plt.tight_layout()
plt.savefig("/home/agi/yuxiaoq/workspace/open-r1-hint/reasoning/distribution.png")
