"""Plot training loss, validation loss, and validation macro-F1."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "output" / "bert_emotion_clf" / "trainer_state.json"
IMAGES_DIR = ROOT / "images"


def main() -> None:
    if not STATE_FILE.is_file():
        raise FileNotFoundError(f"Missing {STATE_FILE}; run train_bert.py first.")
    history = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("log_history", [])
    if not history:
        raise ValueError("No log_history in trainer_state.json")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, series, title, ylabel in (
        ("loss_curve.png", (("loss", "Training"), ("eval_loss", "Validation")), "Loss", "Loss"),
        ("f1_curve.png", (("eval_f1", "Validation macro-F1"),), "Validation macro-F1", "F1"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5))
        plotted = False
        for key, label in series:
            points = [(row["step"], row[key]) for row in history if "step" in row and key in row]
            if points:
                x, y = zip(*points)
                axis.plot(x, y, marker="o" if key.startswith("eval_") else None, label=label)
                plotted = True
        if not plotted:
            plt.close(figure)
            raise ValueError(f"No {title} records in trainer_state.json")
        axis.set(xlabel="Optimizer step", ylabel=ylabel, title=title)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(IMAGES_DIR / filename, dpi=150)
        plt.close(figure)
        print(f"Saved: {IMAGES_DIR / filename}")


if __name__ == "__main__":
    main()
