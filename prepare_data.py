"""Clean the CSV and create stable stratified train/validation/test files."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw.csv"
DATA_DIR = ROOT / "data" / "bert"
MAP_FILE = ROOT / "data" / "label_map.json"
SEED = 42


def write_jsonl(path: Path, frame: pd.DataFrame, label2id: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame.itertuples(index=False):
            record = {"sentences": row.text, "labels": label2id[row.emotion]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Recreate all split files")
    args = parser.parse_args()
    if not RAW_FILE.is_file():
        raise FileNotFoundError(f"Run download_data.py first: {RAW_FILE}")

    expected = [MAP_FILE, *(DATA_DIR / f"{name}.jsonl" for name in ("train", "valid", "test"))]
    if any(path.exists() for path in expected) and not args.force:
        raise FileExistsError("Split files already exist. Use --force only when intentionally resplitting.")

    df = pd.read_csv(RAW_FILE, encoding="utf-8-sig", dtype=str)
    if not {"text", "emotion"}.issubset(df.columns):
        raise ValueError("CSV must contain text and emotion columns")
    df = df[["text", "emotion"]].dropna().copy()
    df["text"] = df["text"].str.strip()
    df["emotion"] = df["emotion"].str.strip()
    df = df[(df["text"] != "") & (df["emotion"] != "")]
    # Conflicting annotations cannot be learned consistently; discard those texts.
    conflicts = df.groupby("text")["emotion"].nunique()
    ambiguous = set(conflicts[conflicts > 1].index)
    df = df[~df["text"].isin(ambiguous)].drop_duplicates(subset="text")
    labels = sorted(df["emotion"].unique().tolist())
    if len(labels) != 8:
        raise ValueError(f"Expected 8 emotion labels, found {len(labels)}: {labels}")
    counts = Counter(df["emotion"])
    if min(counts.values()) < 10:
        raise ValueError("Each label needs at least 10 examples for a 80/10/10 split")
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {str(index): label for label, index in label2id.items()}

    # 20% holdout, then divide that holdout equally for validation and test.
    train, holdout = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["emotion"]
    )
    valid, test = train_test_split(
        holdout, test_size=0.5, random_state=SEED, stratify=holdout["emotion"]
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        write_jsonl(DATA_DIR / f"{name}.jsonl", split, label2id)
        print(f"{name}: {len(split)}; {dict(sorted(Counter(split['emotion']).items()))}")
    MAP_FILE.write_text(
        json.dumps({"label2id": label2id, "id2label": id2label, "seed": SEED}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Cleaned rows: {len(df)}; ambiguous texts removed: {len(ambiguous)}")
    print(f"Labels ({len(labels)}): {labels}")


if __name__ == "__main__":
    main()
