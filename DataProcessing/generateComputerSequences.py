
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
NUM_SEQUENCES = 5000
SEQUENCE_LENGTH = 10
MIN_NUMBER = 1
MAX_NUMBER = 9

# Output path
data_dir = Path(r"D:\personal projects\humanVsComputerGeneration\Data")
output_file = data_dir / "computer_sequences.csv"

# Generate computer sequences
rng = np.random.default_rng(seed=42)

sequences = rng.integers(
    low=MIN_NUMBER,
    high=MAX_NUMBER + 1,
    size=(NUM_SEQUENCES, SEQUENCE_LENGTH)
)

# Create DataFrame with 10 number columns
columns = [f"number_{i}" for i in range(1, SEQUENCE_LENGTH + 1)]
df = pd.DataFrame(sequences, columns=columns)

# Label: 0 = computer-generated
df["label"] = 0

# Save to CSV
df.to_csv(output_file, index=False)

print(f"Generated {NUM_SEQUENCES} computer sequences.")
print(f"Saved to: {output_file}")
print("\nFirst 5 sequences:")
print(df.head())
print(f"\nDataset shape: {df.shape}")