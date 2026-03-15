from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class IntentClassifier:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.data_path = self.base_dir / 'data' / 'intents.csv'
        self.model_path = self.base_dir / 'models' / 'intent_model.pkl'
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            data = pd.read_csv(self.data_path)
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression(max_iter=300)),
            ])
            self.model.fit(data['query'], data['intent'])
            joblib.dump(self.model, self.model_path)

    def predict(self, text: str) -> str:
        return str(self.model.predict([text])[0])
