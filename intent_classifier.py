import json
import re

INTENTS_FILE = r"C:\Users\kamal\OneDrive\Desktop\BotTrainer\data\intents.json"

with open(INTENTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# -------- ENTITY MAP --------
ENTITY_MAP = {}
for entity, values in data.get("entities", {}).items():
    for v in values:
        ENTITY_MAP[v.lower()] = entity


def classify_intent(user_input):
    user_input_lower = user_input.lower()

    best_intent = "unknown"
    best_score = 0

    # ---------- INTENT MATCHING ----------
    for intent in data["intents"]:
        score = 0
        for example in intent["examples"]:
            example_words = example.lower().split()
            for word in example_words:
                if word in user_input_lower:
                    score += 1

        if score > best_score:
            best_score = score
            best_intent = intent["name"]

    confidence = round(min(1.0, best_score / 5), 2)

    # ---------- ENTITY EXTRACTION ----------
    entities = {}

    # Time
    time_match = re.search(r"\b\d{1,2}\s?(am|pm)\b", user_input_lower)
    if time_match:
        entities["reminder_time"] = time_match.group()

    # Numbers
    amount_match = re.search(r"\b\d+\b", user_input_lower)
    if amount_match:
        entities["amount"] = amount_match.group()

    # Dictionary entities
    for value, entity in ENTITY_MAP.items():
        if re.search(r"\b" + re.escape(value) + r"\b", user_input_lower):
            entities[entity] = value

    return {
        "intent": best_intent,
        "confidence": confidence,
        "entities": entities
    }
