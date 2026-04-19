from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re


class Retriever:
    def __init__(self, wiki):
        self.wiki = wiki

        print("🔍 Building TF-IDF index over page titles...")
        self.pages = list(wiki.keys())
        self.page_titles = [p.replace("_", " ").lower() for p in self.pages]

        self.title_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.title_matrix = self.title_vectorizer.fit_transform(self.page_titles)

        print("🤖 Loading Sentence-BERT model...")
        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"✅ Indexed {len(self.pages)} pages.")

    # --------------------------
    # HELPERS
    # --------------------------
    def extract_years(self, text):
        return set(re.findall(r"\b(?:19|20)\d{2}\b", text))

    def normalize(self, text):
        return text.lower().strip()

    # --------------------------
    # PAGE RETRIEVAL
    # --------------------------
    def retrieve_pages(self, claim, top_k=5):

        claim = self.normalize(claim)
        claim_years = self.extract_years(claim)

        claim_vec = self.title_vectorizer.transform([claim])
        scores = cosine_similarity(claim_vec, self.title_matrix)[0]

        boosted_scores = []

        for i, score in enumerate(scores):
            title = self.page_titles[i]
            boost = 0.0

            # 🔥 year boost
            page_years = self.extract_years(title)
            if claim_years and page_years and (claim_years & page_years):
                boost += 0.25

            # 🔥 keyword overlap boost
            if any(word in title for word in claim.split()[:5]):
                boost += 0.05

            boosted_scores.append(score + boost)

        top_indices = np.argsort(boosted_scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            if boosted_scores[i] > 0.015:  # lowered threshold
                results.append({
                    "page": self.pages[i],
                    "score": float(boosted_scores[i])
                })

        return results

    # --------------------------
    # SENTENCE RETRIEVAL
    # --------------------------
    def retrieve_sentences(self, claim, pages, top_k=5):

        all_sentences = []

        for p in pages:
            page = p["page"]
            if page not in self.wiki:
                continue

            for sid, text in self.wiki[page].items():
                if len(text.split()) < 5:
                    continue

                all_sentences.append({
                    "page": page,
                    "sent_id": sid,
                    "text": text
                })

        if not all_sentences:
            return []

        texts = [s["text"] for s in all_sentences]

        # ---------------- TF-IDF FILTER ----------------
        tfidf = TfidfVectorizer(ngram_range=(1, 2))
        mat = tfidf.fit_transform(texts)
        q = tfidf.transform([claim])

        tfidf_scores = cosine_similarity(q, mat)[0]
        top_idx = np.argsort(tfidf_scores)[::-1][:40]

        candidates = [all_sentences[i] for i in top_idx]
        candidate_texts = [c["text"] for c in candidates]

        # ---------------- BERT SEMANTIC ----------------
        claim_emb = self.bert_model.encode([claim], convert_to_numpy=True)
        sent_emb = self.bert_model.encode(candidate_texts, convert_to_numpy=True)

        bert_scores = cosine_similarity(claim_emb, sent_emb)[0]

        # ---------------- HYBRID SCORE ----------------
        final_scores = 0.65 * bert_scores + 0.35 * tfidf_scores[top_idx]

        results = []

        for i, idx in enumerate(top_idx):
            score = final_scores[i]
            text = candidates[i]["text"]

            # 🔥 improved threshold
            if score < 0.15:
                continue

            results.append({
                "page": candidates[i]["page"],
                "sent_id": candidates[i]["sent_id"],
                "text": text,
                "score": float(score)
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # --------------------------
    # MAIN PIPELINE
    # --------------------------
    def retrieve(self, claim, k=5):

        if not claim or not claim.strip():
            return []

        pages = self.retrieve_pages(claim, top_k=10)

        if not pages:
            return []

        return self.retrieve_sentences(claim, pages, top_k=k)