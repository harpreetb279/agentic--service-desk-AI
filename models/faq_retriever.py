from pathlib import Path
import csv
import math
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class FAQRetriever:

    def __init__(self, base_dir: Path):

        self.base_dir = Path(base_dir)

        faq_path = self.base_dir / "data" / "faqs.csv"

        self.questions = []
        self.answers = []

        with open(faq_path, "r", encoding="utf-8") as f:

            reader = csv.reader(f)

            for row in reader:

                if len(row) >= 2:

                    question = row[0].strip()
                    answer = row[1].strip()

                    self.questions.append(question)
                    self.answers.append(answer)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.embeddings = self.embedding_model.encode(self.questions)

        self.backend_name = "hybrid_csv"

    def _cosine_similarity(self, a, b):

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _semantic_scores(self, query_vector):

        scores = []

        for idx, vec in enumerate(self.embeddings):

            score = self._cosine_similarity(query_vector, vec)

            scores.append((idx, score))

        return scores

    def _keyword_score(self, query, question):

        query_tokens = query.lower().split()
        question_tokens = question.lower().split()

        overlap = len(set(query_tokens) & set(question_tokens))

        if overlap == 0:
            return 0.0

        return overlap / math.sqrt(len(question_tokens))

    def _keyword_scores(self, query):

        scores = []

        for idx, q in enumerate(self.questions):

            score = self._keyword_score(query, q)

            scores.append((idx, score))

        return scores

    def search(self, query, limit=3):

        query_vec = self.embedding_model.encode(query)

        semantic = dict(self._semantic_scores(query_vec))
        keyword = dict(self._keyword_scores(query))

        hybrid_scores = []

        for idx in range(len(self.questions)):

            semantic_score = semantic.get(idx, 0)
            keyword_score = keyword.get(idx, 0)

            score = (0.75 * semantic_score) + (0.25 * keyword_score)

            hybrid_scores.append((idx, score))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        top_candidates = hybrid_scores[:10]

        pairs = []

        for idx, _ in top_candidates:

            pairs.append([query, self.questions[idx]])

        rerank_scores = self.rerank_model.predict(pairs)

        reranked = []

        for i, (idx, _) in enumerate(top_candidates):

            reranked.append((idx, float(rerank_scores[i])))

        reranked.sort(key=lambda x: x[1], reverse=True)

        results = []

        for idx, score in reranked[:limit]:

            results.append(
                {
                    "question": self.questions[idx],
                    "answer": self.answers[idx],
                    "score": float(score)
                }
            )

        return results
