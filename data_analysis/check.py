from datasets import load_dataset
import re

# Define the pattern for dataset names
pattern = r"DeepScaleR-1\.5B-Preview_(\d+)_(\d+)"

# Store missing file indices
missing_indices = []

# Check each possible file index between 0 and 140
for file_index in range(141):  # 0 to 140 inclusive
    start_index = file_index * 100
    end_index = start_index + 100
    
    # Construct the dataset name
    dataset_name = f"DeepScaleR-1.5B-Preview_{start_index}_{end_index}"
    
    try:
        # Try to load the dataset with this specific configuration
        dataset = load_dataset('CohenQu/direct_eval_DAPO-Math-en', dataset_name)
        print(f"Found dataset: {dataset_name}")
    except Exception as e:
        # If loading fails, this index is missing
        print(f"Missing dataset: {dataset_name}")
        missing_indices.append(file_index)

# Print the results
print("\nResults:")
print(f"Missing file indices: {sorted(missing_indices)}")
print(f"Total missing: {len(missing_indices)}")