"""Lightweight, deterministic retrieval over the context notes (TF-IDF + cosine similarity).
No network, no model download, no LLM tokens. The hard filters in notes.py still decide verdicts."""
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, notes):
        self.notes = notes.reset_index(drop=True)
        docs = (self.notes["applies_to"] + ". " + self.notes["note"]).tolist()
        self.vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform(docs)
        self.queries = 0
        self.seconds = 0.0

    def search(self, route, week_of, k=3):
        """Top-k notes by similarity to a query about a cost rise on this route."""
        t0 = time.perf_counter()
        q = f"{route} freight cost increase surcharge disruption week of {week_of}"
        sims = cosine_similarity(self.vec.transform([q]), self.matrix)[0]
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], self.notes.loc[i, "note_id"]))[:k]
        self.queries += 1
        self.seconds += time.perf_counter() - t0
        return [(self.notes.loc[i, "note_id"], float(sims[i])) for i in order]

    def score(self, route, week_of):
        """Similarity of every note to the query, as {note_id: score}."""
        q = f"{route} freight cost increase surcharge disruption week of {week_of}"
        sims = cosine_similarity(self.vec.transform([q]), self.matrix)[0]
        return {self.notes.loc[i, "note_id"]: float(sims[i]) for i in range(len(sims))}