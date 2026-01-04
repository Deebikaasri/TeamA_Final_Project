from pathlib import Path
import json
import re

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INTENTS_PATH = DATA_DIR / "intents.json"

with open(INTENTS_PATH, encoding="utf-8") as f:
    INTENT_DATA = json.load(f)

INTENT_KEYWORDS = {
    intent["name"]: intent["examples"]
    for intent in INTENT_DATA["intents"]
}

CITIES = ["chennai", "mumbai", "delhi", "bangalore", "hyderabad"]
FOODS = ["pizza", "biryani", "burger", "sandwich"]
DATES = ["today", "tomorrow", "tonight", "next week"]


def classify_intent(text: str) -> dict:
    text = text.lower()
    best_intent = "unknown"
    best_score = 0

    for intent, examples in INTENT_KEYWORDS.items():
        score = sum(1 for ex in examples if any(w in text for w in ex.lower().split()))
        if score > best_score:
            best_score = score
            best_intent = intent

    confidence = min(1.0, best_score / 5) if best_score else 0.1

    return {"intent": best_intent, "confidence": round(confidence, 2)}


def extract_entities(text: str) -> dict:
    text = text.lower()
    entities = {}

    for c in CITIES:
        if c in text:
            entities["location"] = c.title()

    for f in FOODS:
        if f in text:
            entities["food_item"] = f

    for d in DATES:
        if d in text:
            entities["date"] = d

    time_match = re.search(r"\b\d{1,2}\s?(am|pm)\b", text)
    if time_match:
        entities["time"] = time_match.group()

    return {"entities": entities}
