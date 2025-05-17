import os
import re
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from collections import defaultdict
import numpy as np
import random

def is_repetitive(text, max_repeat_ratio=0.25, min_token_len=2):
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens:
        return False
    counts = defaultdict(int)
    for t in tokens:
        if len(t) >= min_token_len:
            counts[t] += 1
    most_common_freq = max(counts.values(), default=0)
    repeat_ratio = most_common_freq / len(tokens)
    return repeat_ratio > max_repeat_ratio

def build_data(problem, response):
    messages = [
        {
            "role": "user",
            "content": problem
        },
        {
            "role": "assistant",
            "content": response
        }
    ]
    suffix = response
    return {
        "messages": messages,
        "suffix": suffix
    }


# === Parameters ===
dataset_path = "CohenQu/continue_vs_terminate_DAPO-Math-en"
subset_template = "DeepScaleR-1.5B-Preview_{}_{}"
output_dataset_name = "CohenQu/Continue_vs_Terminate.00.00"
output_split = "train"

# === Step 1: Load and Combine Subsets ===
datasets = []
for i in range(0, 100):
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

# === Step 2: Structure Data by Problem ===
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
        data_by_problem[problem].append((progress_val, action, mean_reward, entry))
    except Exception as e:
        print(f"Error processing entry: {e}")
        continue

# === Step 3: Construct Output Dataset ===
processed_data = []
counts = [0, 0, 0]
for problem, entries in data_by_problem.items():
    reward_continue = []
    reward_terminate = []
    terminate_entries = []
    continue_entries = []

    for prog, action, reward, raw in entries:
        if action == "Terminate":
            reward_terminate.append(reward)
            terminate_entries.append((prog, reward, raw))
        elif action == "Continue":
            reward_continue.append(reward)
            continue_entries.append((prog, reward, raw))
    if problem == "At Ignus School, there are $425$ students. Of these students, $351$ study mathematics, $71$ study Latin, and $203$ study chemistry. There are $199$ students who study more than one of these subjects, and $8$ students who do not study any of these subjects. Find the number of students who study all three of these subjects.":
        continue
    
    init_response = raw["init_response"]
    
    if is_repetitive(init_response, max_repeat_ratio=0.25, min_token_len=2):
        continue 

    if all(r == 0 for r in reward_continue + reward_terminate):
        counts[0] += 1
        continue  # Skip problem

    # Try to find earliest point where terminate > continue at same progress
    success = False
    for i in range(min(len(terminate_entries), len(continue_entries))):
        if terminate_entries[i][0] == continue_entries[i][0]:  # match progress
            if terminate_entries[i][1] > 0 and terminate_entries[i][1] >= continue_entries[i][1]:
                print(f"terminate_reward ({terminate_entries[i][1]}) >= continue_reward ({continue_entries[i][1]})")
                chosen_raw = terminate_entries[i][2]
                responses = chosen_raw["responses"]
                rewards = chosen_raw["rewrads"]
                
                if not responses or not rewards:
                    break
                max_reward = max(rewards)
                candidates = [(r, len(r)) for r, rw in zip(responses, rewards) if rw == max_reward]
                best_response = sorted(candidates, key=lambda x: x[1])[0][0]
                
                response = chosen_raw["prefix"].split("Time is up")[0] + "Continuing to generate further reasoning offers no additional benefit for this problem. So I will now directly produce the final answer based on the inferences already made.\n</think>\n" + best_response.strip()
                if "<think>" not in response:
                    response = "<think>\n" + response
                processed_data.append(build_data(chosen_raw["problem"], response))
                success = True
                break
        else:
            print("progress doesn't math")

    if not success:
        # Fallback: use init_response if no progress point found
        counts[2] += 1
        fallback = entries[0][3]
        response = fallback["init_response"]
        if "<think>" not in response:
            response = "<think>\n" + response
        processed_data.append(build_data(fallback["problem"], response))
    else:
        counts[1] += 1

print(counts)

# === Step 4: Upload Dataset ===
random.seed(42)
random.shuffle(processed_data)

eval_data = processed_data[-50:]
train_data = processed_data[:-50]

# Create and upload
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)
final_dataset = DatasetDict({
    "train": train_dataset,
    "test": eval_dataset
})
# final_dataset.push_to_hub(output_dataset_name)