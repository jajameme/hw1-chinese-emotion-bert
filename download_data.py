"""Download the original CSV, or copy an already downloaded CSV into data/."""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "data" / "raw.csv"
URL = (
    "https://huggingface.co/datasets/Johnson8187/"
    "Chinese_Multi-Emotion_Dialogue_Dataset/resolve/main/data.csv"
)


def summarize(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"text", "emotion"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain text and emotion columns")
        counts = Counter(row["emotion"].strip() for row in reader)
    print(f"Raw rows: {sum(counts.values())}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Use a manually downloaded data.csv")
    parser.add_argument("--force", action="store_true", help="Replace existing raw.csv")
    args = parser.parse_args()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists() and not args.force:
        print(f"Keeping existing {DESTINATION}")
        summarize(DESTINATION)
        return

    temporary = DESTINATION.with_suffix(".csv.tmp")
    try:
        if args.source:
            if args.source.resolve() == DESTINATION.resolve():
                raise ValueError("Source and destination are the same file")
            shutil.copyfile(args.source, temporary)
        else:
            with urlopen(URL, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
        summarize(temporary)
        temporary.replace(DESTINATION)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved: {DESTINATION}")


if __name__ == "__main__":
    main()
