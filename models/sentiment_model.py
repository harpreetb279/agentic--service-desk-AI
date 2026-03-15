from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class SentimentClassifier:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.data_path = self.base_dir / 'data' / 'sentiment.csv'
        self.model_path = self.base_dir / 'models' / 'sentiment_model.pkl'
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            data = pd.read_csv(self.data_path)
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', MultinomialNB()),
            ])
            self.model.fit(data['query'], data['sentiment'])
            joblib.dump(self.model, self.model_path)

    def predict(self, text: str) -> str:
        return str(self.model.predict([text])[0])
