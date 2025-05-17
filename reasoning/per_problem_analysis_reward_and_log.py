import os
import re
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# === Parameters ===
dataset_path = "CohenQu/continue_vs_terminate_DAPO-Math-en"
subset_template = "DeepScaleR-1.5B-Preview_{}_{}"
output_dir = "/home/agi/yuxiaoq/workspace/open-r1-hint/reasoning/space_terminate"
os.makedirs(output_dir, exist_ok=True)

# === Load model and tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "agentica-org/DeepScaleR-1.5B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

def compute_close_think_logprob(prompt: str) -> float:
    """Returns log prob of '</think>' given the prompt."""
    close_token = "\n</think>"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_input = tokenizer(prompt + close_token, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output = model(full_input)
        logits = output.logits

    # Get the token id for </think>
    close_token_id = tokenizer.convert_tokens_to_ids(close_token)
    # Index of the next token to predict is the last one
    log_probs = torch.nn.functional.log_softmax(logits[:, -2, :], dim=-1)
    return log_probs[0, close_token_id].item()

# === Step 1: Load and Combine Subsets ===
datasets = []
for i in range(0, 1):
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

for entry in tqdm(combined, desc="Parsing entries"):
    try:
        problem = entry["problem"]
        progress_match = re.match(r"(\d+)\s*/\s*(\d+)", entry["progress"])
        if not progress_match:
            continue
        num, denom = int(progress_match.group(1)), int(progress_match.group(2))
        progress_val = num / denom if denom != 0 else 0.0
        action = entry["action"]
        mean_reward = entry["mean_reward"]
        prompt = entry["tokenized_conv"]
        
        close_think_logprob = compute_close_think_logprob(prompt) if action == "Continue" else None
        data_by_problem[problem].append((progress_val, action, mean_reward, close_think_logprob))
    except Exception as e:
        print(f"Error processing entry: {e}")
        continue

# === Step 3: Plot per Problem ===
for idx, (problem, entries) in enumerate(sorted(data_by_problem.items())):
    progress_terminate = []
    reward_terminate = []
    progress_continue = []
    reward_continue = []
    logprob_progress = []
    logprob_values = []

    for prog, action, reward, logprob in entries:
        if action == "Terminate":
            progress_terminate.append(prog)
            reward_terminate.append(reward)
        elif action == "Continue":
            progress_continue.append(prog)
            reward_continue.append(reward)
            if logprob is not None:
                logprob_progress.append(prog)
                logprob_values.append(logprob)

    fig, ax1 = plt.subplots()

    # Plot rewards
    if progress_terminate:
        x_t, y_t = zip(*sorted(zip(progress_terminate, reward_terminate)))
        ax1.plot(x_t, y_t, label="Terminate", marker="o", linestyle="-")
    if progress_continue:
        x_c, y_c = zip(*sorted(zip(progress_continue, reward_continue)))
        ax1.plot(x_c, y_c, label="Continue", marker="x", linestyle="--")

    ax1.set_xlabel("Progress (int1 / int2)")
    ax1.set_ylabel("Mean Reward")
    # ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")

    # Plot log-probs on second y-axis
    if logprob_progress:
        ax2 = ax1.twinx()
        x_l, y_l = zip(*sorted(zip(logprob_progress, logprob_values)))
        ax2.plot(x_l, y_l, label="LogP('</think>')", marker=".", color="gray")
        ax2.set_ylabel("LogP('</think>')")
        ax2.legend(loc="upper right")

    plt.title(f"Problem {idx}")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{idx}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot for problem {idx} to {save_path}")