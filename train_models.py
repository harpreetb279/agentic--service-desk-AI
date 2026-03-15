from pathlib import Path
from models.intent_model import IntentClassifier
from models.sentiment_model import SentimentClassifier

base_dir = Path(__file__).resolve().parent
IntentClassifier(base_dir=base_dir)
SentimentClassifier(base_dir=base_dir)
print('training complete')
