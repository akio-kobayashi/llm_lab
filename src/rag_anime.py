import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


class AnimeNewsRAG:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.docs: List[Dict] = []
        self.embeddings = None

    def load_jsonl(self, path: str):
        self.docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.docs.append(json.loads(line))

    def build_index(self):
        texts = [f"{d.get('title','')}\n{d.get('summary','')}" for d in self.docs]
        if not texts:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            return
        emb = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb.astype(np.float32)

    def search(self, query: str, top_k: int = 3):
        if self.embeddings is None or len(self.docs) == 0:
            return []
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        scores = self.embeddings @ q
        idx = np.argsort(-scores)[:top_k]
        out = []
        for i in idx:
            d = dict(self.docs[int(i)])
            d["score"] = float(scores[int(i)])
            out.append(d)
        return out


def ensure_anime_news_file(path: str):
    p = Path(path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    seed = {
        "title": "2026年2月アニメ情報の収集テンプレート",
        "date": "2026-02-01",
        "source": "https://animeanime.jp/article/2026/02/",
        "summary": "このファイルに2026年2月の最新記事を追加してRAG対象にする。"
    }
    p.write_text(json.dumps(seed, ensure_ascii=False) + "\n", encoding="utf-8")
