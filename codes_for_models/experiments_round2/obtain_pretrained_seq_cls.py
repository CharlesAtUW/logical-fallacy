import sys
import os
from transformers import AutoModelForSequenceClassification

DEFAULT_MODEL = "howey/electra-small-mnli"
SAVED_MODELS_LOCATION = "../../saved_models"

def save_pretrained_sequence_classification_model(model_name: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(os.path.join(SAVED_MODELS_LOCATION, os.path.split(model_name)[1]))


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_MODEL
    save_pretrained_sequence_classification_model(model_name)