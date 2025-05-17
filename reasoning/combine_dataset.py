from datasets import load_dataset, concatenate_datasets

# Step 1: Load and combine subsets
base_path = "CohenQu/direct_eval_DAPO-Math-en"
combined_subset_name = "DeepScaleR-1.5B-Preview_0_2000"

subsets = []
for j in range(20):  # 0 to 9 -> _0_100, _100_200, ..., _900_1000
    subset_name = f"DeepScaleR-1.5B-Preview_{j*100}_{j*100+100}"
    ds = load_dataset(base_path, name=subset_name, split="train")
    subsets.append(ds)

# Step 2: Concatenate all loaded subsets
combined_dataset = concatenate_datasets(subsets)

# Step 3: Push the combined dataset to the same hub with new name
combined_dataset.push_to_hub(
    repo_id=base_path,
    config_name=combined_subset_name,
    split="train"
)
