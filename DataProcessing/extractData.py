# This script extracts the human data from the original dataset and saves it in a new CSV file with 10 separate columns for each number in the sequence.

import pandas as pd
from pathlib import Path

# File paths
data_dir = Path(r"D:\personal projects\humanVsComputerGeneration\Data")
input_file = data_dir / "human_data.csv"
output_file = data_dir / "human_sequences.csv"

# Load the original dataset
df = pd.read_csv(input_file)

# Extract the test trial sequences
sequences = df["number_choices_test_trials"].dropna()

# Split each sequence into 10 separate numbers
split_sequences = sequences.apply(
    lambda x: [int(n.strip()) for n in str(x).split(",")]
)

# Verify that every sequence contains exactly 10 numbers
invalid = split_sequences[split_sequences.apply(len) != 10]

if not invalid.empty:
    raise ValueError(
        f"Found {len(invalid)} sequences that do not contain exactly 10 numbers."
    )

# Create a DataFrame with 10 columns
columns = [f"number_{i}" for i in range(1, 11)]
output_df = pd.DataFrame(split_sequences.tolist(), columns=columns)

# Add labels: 1 = human-generated, 0 = computer-generated
output_df["label"] = 1

# Save to CSV
output_df.to_csv(output_file, index=False)

print(f"Successfully saved {len(output_df)} sequences.")
print(f"Output file: {output_file}")
print(output_df.head())