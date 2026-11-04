"""Run the complete sequence data and modeling pipeline from the project root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
MODEL_DIR = PROJECT_ROOT / "Models"
METRICS_DIR = PROJECT_ROOT / "Reports"
SEQUENCE_COLUMNS = [f"number_{index}" for index in range(1, 11)]
LABEL_NAMES = {0: "Computer", 1: "Human"}


def _add_features(sequences: np.ndarray) -> pd.DataFrame:
    """Extract order-aware summaries and value-frequency features."""
    rows: list[dict[str, float | int]] = []
    for sequence in sequences:
        x = np.asarray(sequence, dtype=int)
        differences = np.diff(x)
        gaps = np.abs(differences)
        values, counts = np.unique(x, return_counts=True)
        frequencies = counts / len(x)
        directions = np.sign(differences)
        nonzero_directions = directions[directions != 0]

        def longest_run(mask: np.ndarray) -> int:
            best = current = 1
            for continues in mask:
                current = current + 1 if continues else 1
                best = max(best, current)
            return best

        features: dict[str, float | int] = {
            "unique_count": len(values),
            "repeat_count": len(x) - len(values),
            "unique_ratio": len(values) / len(x),
            "max_frequency": int(counts.max()),
            "single_occurrence_count": int(np.sum(counts == 1)),
            "repeated_value_count": int(np.sum(counts > 1)),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "std": float(np.std(x)),
            "minimum": int(np.min(x)),
            "maximum": int(np.max(x)),
            "value_range": int(np.ptp(x)),
            "adjacent_repeat_count": int(np.sum(differences == 0)),
            "longest_repeat_run": longest_run(differences == 0),
            "mean_adjacent_difference": float(np.mean(gaps)),
            "std_adjacent_difference": float(np.std(gaps)),
            "max_adjacent_difference": int(np.max(gaps)),
            "ascending_pairs": int(np.sum(differences == 1)),
            "descending_pairs": int(np.sum(differences == -1)),
            "increasing_pairs": int(np.sum(differences > 0)),
            "decreasing_pairs": int(np.sum(differences < 0)),
            "small_gap_pairs": int(np.sum(gaps <= 2)),
            "longest_increasing_run": longest_run(differences > 0),
            "longest_decreasing_run": longest_run(differences < 0),
            "direction_changes": int(np.sum(nonzero_directions[1:] != nonzero_directions[:-1])),
            "entropy": float(-np.sum(frequencies * np.log2(frequencies))),
            "odd_count": int(np.sum(x % 2 == 1)),
            "even_count": int(np.sum(x % 2 == 0)),
            "parity_changes": int(np.sum((x % 2)[1:] != (x % 2)[:-1])),
            "lower_range_count": int(np.sum(x <= 5)),
            "upper_range_count": int(np.sum(x >= 6)),
            "gap_unique_count": len(np.unique(gaps)),
            "gap_repeat_count": len(gaps) - len(np.unique(gaps)),
            "max_gap_frequency": int(np.unique(gaps, return_counts=True)[1].max()),
            "distinct_adjacent_pairs": len(set(zip(x[:-1], x[1:]))),
            "local_peaks": sum(x[i] > x[i - 1] and x[i] > x[i + 1] for i in range(1, len(x) - 1)),
            "local_valleys": sum(x[i] < x[i - 1] and x[i] < x[i + 1] for i in range(1, len(x) - 1)),
        }
        features.update({f"count_{number}": int(np.sum(x == number)) for number in range(1, 10)})
        rows.append(features)
    return pd.DataFrame(rows)


def run_pipeline(seed: int = 42, computer_count: int = 5_000) -> dict[str, object]:
    """Regenerate derived data, train a held-out baseline, and save artifacts."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from DataProcessing.combine_datasets import combine_datasets
    from DataProcessing.extractData import extract_human_sequences
    from DataProcessing.generateComputerSequences import generate_computer_sequences

    human_file = DATA_DIR / "human_sequences.csv"
    computer_file = DATA_DIR / "computer_sequences.csv"
    combined_file = DATA_DIR / "combined_dataset.csv"
    clean_file = DATA_DIR / "clean_combined_dataset.csv"
    features_file = DATA_DIR / "engineered_features.csv"
    for directory in (DATA_DIR, MODEL_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    human = extract_human_sequences()
    computer = generate_computer_sequences(count=computer_count, seed=seed)
    human.to_csv(human_file, index=False)
    computer.to_csv(computer_file, index=False)
    combined = combine_datasets(human_file, computer_file)
    duplicate_count = int(combined.duplicated().sum())
    clean = combined.drop_duplicates().reset_index(drop=True)
    clean.to_csv(combined_file, index=False)
    clean.to_csv(clean_file, index=False)

    X = _add_features(clean[SEQUENCE_COLUMNS].to_numpy())
    y = clean["label"].astype(int)
    feature_table = X.copy()
    feature_table["label"] = y
    feature_table.to_csv(features_file, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    metrics = {
        "label_mapping": {str(label): name for label, name in LABEL_NAMES.items()},
        "seed": seed,
        "computer_sequences_requested": computer_count,
        "human_sequences": int((clean.label == 1).sum()),
        "computer_sequences": int((clean.label == 0).sum()),
        "duplicate_rows_removed": duplicate_count,
        "feature_count": int(X.shape[1]),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "human_precision": float(precision_score(y_test, prediction, pos_label=1, zero_division=0)),
        "human_recall": float(recall_score(y_test, prediction, pos_label=1, zero_division=0)),
        "human_f1": float(f1_score(y_test, prediction, pos_label=1, zero_division=0)),
        "roc_auc_human_positive": float(roc_auc_score(y_test, probability)),
        "confusion_matrix_labels_0_computer_1_human": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, prediction, labels=[0, 1], target_names=["Computer", "Human"],
            output_dict=True, zero_division=0,
        ),
        "interpretation_limit": (
            "This score distinguishes the observed participant sample from this seeded uniform 1-9 generator; "
            "it does not establish performance against every possible computer generator."
        ),
    }
    joblib.dump({"model": model, "feature_columns": list(X.columns), "label_names": LABEL_NAMES}, MODEL_DIR / "random_forest.joblib")
    (METRICS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation and train/test split.")
    parser.add_argument("--computer-count", type=int, default=5_000, help="Number of computer sequences to generate.")
    args = parser.parse_args()
    metrics = run_pipeline(seed=args.seed, computer_count=args.computer_count)
    print(f"Pipeline complete: {metrics['human_sequences']:,} human + {metrics['computer_sequences']:,} computer sequences")
    print(f"Accuracy: {metrics['accuracy']:.3f} | Human F1: {metrics['human_f1']:.3f} | ROC-AUC: {metrics['roc_auc_human_positive']:.3f}")
    print(f"Features: {DATA_DIR / 'engineered_features.csv'}")
    print(f"Model: {MODEL_DIR / 'random_forest.joblib'}")
    print(f"Metrics: {METRICS_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
