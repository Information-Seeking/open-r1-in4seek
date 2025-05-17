from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

# === Parameters ===
dataset_path = "CohenQu/continue_vs_termination_AIME2025"
subset_names = [
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepScaleR-1.5B-Preview",
    "Qwen3-1.7B",
    "Qwen3-1.7B_Continue_vs_Terminate.00.00_5e-5",
    "DeepScaleR-1.5B-Preview_Continue_vs_Terminate.00.00_5e-5",
    "DeepSeek-R1-Distill-Qwen-1.5B_Continue_vs_Terminate.00.00_5e-5",
]

model_maps = {
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "DeepScaleR-1.5B-Preview": "agentica-org/DeepScaleR-1.5B-Preview",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-1.7B_Continue_vs_Terminate.00.00_5e-5": "CohenQu/Qwen3-1.7B_Continue_vs_Terminate.00.00_5e-5",
    "DeepScaleR-1.5B-Preview_Continue_vs_Terminate.00.00_5e-5": "CohenQu/DeepScaleR-1.5B-Preview_Continue_vs_Terminate.00.00_5e-5",
    "DeepSeek-R1-Distill-Qwen-1.5B_Continue_vs_Terminate.00.00_5e-5": "CohenQu/DeepSeek-R1-Distill-Qwen-1.5B_Continue_vs_Terminate.00.00_5e-5",
}



# Collect results
results = {}

for subset in subset_names:
    try:
        ds = load_dataset(dataset_path, name=f"{subset}_0_30", split="AIME2025")
    except Exception as e:
        print(f"Failed to load {subset}: {e}")
        continue

    rewards_all, tokens_all, tokens_r1, tokens_r0 = [], [], [], []
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_maps[subset])
    for row in ds:
        for reward, response in zip(row["rewards"], row["responses"]):
            token_len = len(tokenizer.encode(response, add_special_tokens=False))
            rewards_all.append(reward)
            tokens_all.append(token_len)
            if reward == 1:
                tokens_r1.append(token_len)
            elif reward == 0:
                tokens_r0.append(token_len)

    results[subset] = {
        "success_rate": np.mean(np.array(rewards_all) == 1),
        "avg_tokens_all": np.mean(tokens_all) if tokens_all else 0,
        "avg_tokens_r1": np.mean(tokens_r1) if tokens_r1 else 0,
        "avg_tokens_r0": np.mean(tokens_r0) if tokens_r0 else 0,
    }

# Print results
df = pd.DataFrame(results).T
print(df)
