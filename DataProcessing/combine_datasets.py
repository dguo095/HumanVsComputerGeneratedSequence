
import pandas as pd
from pathlib import Path

# Data directory
data_dir = Path(r"D:\personal projects\humanVsComputerGeneration\Data")

# Input files
human_file = data_dir / "human_sequences.csv"
computer_file = data_dir / "computer_sequences.csv"

# Output file
output_file = data_dir / "combined_dataset.csv"

# Load datasets
human_df = pd.read_csv(human_file)
computer_df = pd.read_csv(computer_file)

# Combine datasets
combined_df = pd.concat(
    [human_df, computer_df],
    ignore_index=True
)

# Shuffle rows reproducibly
combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save combined dataset
combined_df.to_csv(output_file, index=False)

# Display summary
print("Datasets combined successfully!")
print(f"Human sequences: {len(human_df)}")
print(f"Computer sequences: {len(computer_df)}")
print(f"Total sequences: {len(combined_df)}")
print(f"\nLabel distribution:\n{combined_df['label'].value_counts().sort_index()}")
print(f"\nOutput saved to: {output_file}")
print(f"Dataset shape: {combined_df.shape}")