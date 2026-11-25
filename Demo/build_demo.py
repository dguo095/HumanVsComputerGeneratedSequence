"""Embed the fitted baseline model in the standalone browser demo."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
MODEL_PATH = DEMO_DIR.parent / "Models" / "random_forest.joblib"
TEMPLATE_PATH = DEMO_DIR / "human-or-computer.template.html"
OUTPUT_PATH = DEMO_DIR / "human-or-computer.html"
NODE = struct.Struct("<hHd")


def export_forest(artifact: dict) -> dict:
    model = artifact["model"]
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected model classes 0 (Computer) and 1 (Human).")

    roots = []
    records = bytearray()
    for estimator in model.estimators_:
        tree = estimator.tree_
        roots.append(len(records) // NODE.size)
        for index, feature in enumerate(tree.feature):
            left = tree.children_left[index]
            if left < 0:
                computer, human = tree.value[index, 0]
                records.extend(NODE.pack(-1, 0, human / (human + computer)))
                continue

            # Each tree is stored in preorder. Its left child follows its parent.
            right_offset = tree.children_right[index] - index
            if left != index + 1 or not 1 <= right_offset <= 0xFFFF:
                raise ValueError("Tree structure is incompatible with the demo format.")
            records.extend(NODE.pack(feature, right_offset, tree.threshold[index]))

    return {
        "columns": artifact["feature_columns"],
        "roots": roots,
        "nodes": base64.b64encode(records).decode("ascii"),
    }


def main() -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit("Model file missing. Run python DataProcessing/pipeline.py first.")

    import joblib

    forest = export_forest(joblib.load(MODEL_PATH))
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    payload = json.dumps(forest, separators=(",", ":"))
    OUTPUT_PATH.write_text(template.replace("__FOREST_DATA__", payload), encoding="utf-8", newline="\n")
    print(f"Built {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
