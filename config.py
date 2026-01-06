import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")
EVAL_DATASET_FILE = os.path.join(DATA_DIR, "eval_dataset.json")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
