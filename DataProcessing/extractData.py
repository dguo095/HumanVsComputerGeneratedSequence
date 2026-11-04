"""Extract the ten test-trial choices from the human participant data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
INPUT_FILE = DATA_DIR / "human_data.csv"
OUTPUT_FILE = DATA_DIR / "human_sequences.csv"
SEQUENCE_COLUMN = "number_choices_test_trials"
SEQUENCE_LENGTH = 10


def extract_human_sequences(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Return validated human sequences using label 1 for human."""
    data = pd.read_csv(input_file)
    if SEQUENCE_COLUMN not in data:
        raise ValueError(f"Required column {SEQUENCE_COLUMN!r} is missing from {input_file}.")

    sequences = data[SEQUENCE_COLUMN].dropna().astype(str)
    parsed = []
    for row_number, sequence in sequences.items():
        try:
            values = [int(value.strip()) for value in sequence.split(",")]
        except ValueError as exc:
            raise ValueError(f"Invalid sequence at input row {row_number + 2}.") from exc
        if len(values) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected {SEQUENCE_LENGTH} values at input row {row_number + 2}; found {len(values)}."
            )
        if any(value < 1 or value > 9 for value in values):
            raise ValueError(f"Values must be between 1 and 9 at input row {row_number + 2}.")
        parsed.append(values)

    columns = [f"number_{index}" for index in range(1, SEQUENCE_LENGTH + 1)]
    result = pd.DataFrame(parsed, columns=columns)
    result["label"] = 1
    return result


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result = extract_human_sequences()
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(result):,} human sequences to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
