"""Fine-tune Chinese BERT and evaluate the selected model once on the test set."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "bert"
OUTPUT_DIR = ROOT / "output" / "bert_emotion_clf"
IMAGES_DIR = ROOT / "images"
MODEL_NAME = "google-bert/bert-base-chinese"
MAX_SEQ_LENGTH = 128
SEED = 42
NUM_TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
EVAL_STEPS = 50
USE_FP16 = True  # Set to False if training loss becomes NaN on GTX 1660 Ti.


def compute_metrics(result):
    logits = result.predictions[0] if isinstance(result.predictions, tuple) else result.predictions
    predicted = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(result.label_ids, predicted),
        "f1": f1_score(result.label_ids, predicted, average="macro", zero_division=0),
    }


def save_confusion_matrix(actual, predicted, names: list[str]) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    font_path = ROOT.parent / "fonts" / "NotoSansTC-Regular.otf"
    if font_path.is_file():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(names))))
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Test confusion matrix")
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(
                column, row, str(matrix[row, column]), ha="center", va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    figure.tight_layout()
    figure.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    (OUTPUT_DIR / "confusion_matrix.json").write_text(
        json.dumps({"labels": names, "matrix": matrix.tolist()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    mapping_path = ROOT / "data" / "label_map.json"
    files = {name: DATA_DIR / f"{filename}.jsonl" for name, filename in (
        ("train", "train"), ("validation", "valid"), ("test", "test")
    )}
    for path in [mapping_path, *files.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run download_data.py and prepare_data.py first.")
    if (OUTPUT_DIR / "model.safetensors").exists():
        raise FileExistsError(f"Model already exists at {OUTPUT_DIR}; archive it before retraining.")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    label2id = mapping["label2id"]
    id2label = {int(index): label for index, label in mapping["id2label"].items()}
    names = [id2label[index] for index in range(len(id2label))]
    if len(names) != 8 or len(names) != len(label2id) or set(names) != set(label2id):
        raise ValueError("Invalid label_map.json")

    raw = load_dataset("json", data_files={name: str(path) for name, path in files.items()})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = raw.map(
        lambda batch: tokenizer(batch["sentences"], truncation=True, max_length=MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=["sentences"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(names), id2label=id2label, label2id=label2id
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=30,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=USE_FP16 and torch.cuda.is_available(),
        bf16=False,
        tf32=False,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    # Test data is held out from model selection and is evaluated only here.
    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    test_logits = (
        test_output.predictions[0]
        if isinstance(test_output.predictions, tuple)
        else test_output.predictions
    )
    predicted = np.argmax(test_logits, axis=-1)
    actual = test_output.label_ids
    report = classification_report(
        actual, predicted, labels=list(range(len(names))), target_names=names,
        output_dict=True, zero_division=0,
    )
    report["test_metrics"] = test_output.metrics
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    trainer.save_state()
    (OUTPUT_DIR / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_confusion_matrix(actual, predicted, names)
    for checkpoint in OUTPUT_DIR.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
    print(f"Test accuracy: {test_output.metrics['test_accuracy']:.4f}")
    print(f"Test macro-F1: {test_output.metrics['test_f1']:.4f}")
    print(f"Best model and report: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
