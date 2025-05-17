import os
import re
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
import numpy as np

# === Parameters ===
dataset_path = "CohenQu/continue_vs_terminate_DAPO-Math-en"
subset_template = "DeepScaleR-1.5B-Preview_{}_{}"
output_dir = "/home/agi/yuxiaoq/workspace/open-r1-hint/reasoning/output"
os.makedirs(output_dir, exist_ok=True)

# === Step 1: Load and Combine Subsets ===
datasets = []
for i in range(0, 20):
    name = subset_template.format(i * 20, i * 20 + 20)
    try:
        subset = load_dataset(dataset_path, name=name, split="test")
        datasets.append(subset)
        print(f"Loaded subset: {name}")
    except Exception as e:
        print(f"Skipping subset {name}: {e}")
        continue

if not datasets:
    raise RuntimeError("No subsets could be loaded.")

combined = concatenate_datasets(datasets)

# === Step 2: Parse Progress and Structure Data ===
data_by_problem = defaultdict(list)

for entry in combined:
    try:
        problem = entry["problem"]
        progress_match = re.match(r"(\d+)\s*/\s*(\d+)", entry["progress"])
        if not progress_match:
            continue
        num, denom = int(progress_match.group(1)), int(progress_match.group(2))
        progress_val = num / denom if denom != 0 else 0.0
        action = entry["action"]
        mean_reward = entry["mean_reward"]
        data_by_problem[problem].append((progress_val, action, mean_reward))
    except Exception as e:
        print(f"Error processing entry: {e}")
        continue

# === Step 3: Plot per Problem ===
for idx, (problem, entries) in enumerate(data_by_problem.items()):
    progress_terminate = []
    reward_terminate = []
    progress_continue = []
    reward_continue = []

    for prog, action, reward in entries:
        if action == "Terminate":
            progress_terminate.append(prog)
            reward_terminate.append(reward)
        elif action == "Continue":
            progress_continue.append(prog)
            reward_continue.append(reward)

    plt.figure()
    if progress_terminate:
        sorted_t = sorted(zip(progress_terminate, reward_terminate))
        x_t, y_t = zip(*sorted_t)
        plt.plot(x_t, y_t, label="Terminate", marker="o")

    if progress_continue:
        sorted_c = sorted(zip(progress_continue, reward_continue))
        x_c, y_c = zip(*sorted_c)
        plt.plot(x_c, y_c, label="Continue", marker="x")

    plt.xlabel("Progress (int1 / int2)")
    plt.ylabel("Mean Reward")
    plt.title(f"Problem {idx}")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{idx}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot for problem {idx} to {save_path}")