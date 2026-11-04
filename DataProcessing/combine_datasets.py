"""Combine the labeled human and computer sequence tables."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
HUMAN_FILE = DATA_DIR / "human_sequences.csv"
COMPUTER_FILE = DATA_DIR / "computer_sequences.csv"
OUTPUT_FILE = DATA_DIR / "combined_dataset.csv"
SEQUENCE_COLUMNS = [f"number_{index}" for index in range(1, 11)]
EXPECTED_COLUMNS = SEQUENCE_COLUMNS + ["label"]


def combine_datasets(
    human_file: Path = HUMAN_FILE, computer_file: Path = COMPUTER_FILE
) -> pd.DataFrame:
    """Load, validate, concatenate, and reproducibly shuffle both sources."""
    human = pd.read_csv(human_file)
    computer = pd.read_csv(computer_file)
    for name, frame, label in (
        ("human", human, 1),
        ("computer", computer, 0),
    ):
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {name} data: {list(frame.columns)}")
        if not frame["label"].eq(label).all():
            raise ValueError(f"{name} data must use label {label}.")
        values = frame[SEQUENCE_COLUMNS]
        if values.isna().any().any() or not values.apply(
            lambda column: column.between(1, 9).all()
        ).all():
            raise ValueError(f"{name} sequences must contain values from 1 to 9.")

    combined = pd.concat([human, computer], ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    result = combine_datasets()
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Combined {len(result):,} sequences: {len(result[result.label == 1]):,} human, "
          f"{len(result[result.label == 0]):,} computer.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
