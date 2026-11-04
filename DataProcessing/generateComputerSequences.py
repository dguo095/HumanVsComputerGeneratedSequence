"""Generate reproducible computer sequences using label 0 for computer."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_FILE = DATA_DIR / "computer_sequences.csv"
NUM_SEQUENCES = 5_000
SEQUENCE_LENGTH = 10
MIN_NUMBER = 1
MAX_NUMBER = 9
RANDOM_SEED = 42


def generate_computer_sequences(
    count: int = NUM_SEQUENCES, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Generate independent uniform draws in the same value range as humans."""
    if count < 1:
        raise ValueError("count must be a positive integer")
    rng = np.random.default_rng(seed)
    values = rng.integers(
        MIN_NUMBER, MAX_NUMBER + 1, size=(count, SEQUENCE_LENGTH)
    )
    columns = [f"number_{index}" for index in range(1, SEQUENCE_LENGTH + 1)]
    result = pd.DataFrame(values, columns=columns)
    result["label"] = 0
    return result


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = generate_computer_sequences()
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result):,} computer sequences to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
