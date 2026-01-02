import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from intent_classifier import classify_intent

def evaluate_model(eval_file="eval_data.json"):
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)["eval_data"]

    true_intents = []
    predicted_intents = []

    for item in eval_data:
        intent_name = item["intent"]
        for example in item["examples"]:
            true_intents.append(intent_name)
            result = classify_intent(example)
            predicted_intents.append(result["intent"])

    accuracy = accuracy_score(true_intents, predicted_intents)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_intents, predicted_intents, average="weighted", zero_division=0
    )

    cm = confusion_matrix(true_intents, predicted_intents)
    labels = sorted(list(set(true_intents)))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "labels": labels
    }
