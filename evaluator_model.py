from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from intent_classifier import classify_intent

BASE_DIR = Path(__file__).parent
EVAL_PATH = BASE_DIR / "data" / "eval_dataset.json"


def run_evaluation():
    with open(EVAL_PATH, encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    y_true = [item["intent"] for item in data]
    y_pred = [classify_intent(t)["intent"] for t in texts]

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
