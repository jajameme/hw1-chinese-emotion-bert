"""Print predictions for the five required homework sentences."""

from pathlib import Path

import torch
from transformers import pipeline


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "output" / "bert_emotion_clf"
TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
]


def main() -> None:
    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"Missing trained model: {MODEL_DIR}. Run train_bert.py first.")
    classifier = pipeline(
        "text-classification", model=str(MODEL_DIR),
        device=0 if torch.cuda.is_available() else -1,
    )
    results = classifier(TEXTS, truncation=True, max_length=128)
    for sentence, result in zip(TEXTS, results, strict=True):
        print(f"{sentence} => {result['label']} ({result['score']:.4f})")


if __name__ == "__main__":
    main()
