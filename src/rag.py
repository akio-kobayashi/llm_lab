import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RagEngine:
    def __init__(self, emb_model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(emb_model_id)
        self.index = None
        self.chunks = []

    def load_documents(self, jsonl_path):
        """
        JSONL形式のドキュメントを読み込む
        """
        self.chunks = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line))
        return len(self.chunks)

    def build_index(self):
        """
        ベクトルインデックスを構築
        """
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # 内積(正規化済みなのでcos類似)
        self.index.add(embeddings.astype('float32'))

    def save_index(self, index_path, chunks_path):
        """
        インデックスを保存
        """
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def load_index(self, index_path, chunks_path):
        """
        インデックスを読み込み
        """
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

    def search(self, query, top_k=3):
        """
        類似チャンクを検索
        """
        query_vector = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_vector.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(distances[0][i])
                })
        return results

def format_rag_prompt(query, results):
    """
    RAGの検索結果をプロンプト形式に変換
    """
    context = ""
    for i, res in enumerate(results):
        context += f"[資料{i+1}]\n{res['chunk']['text']}\n\n"
    
    prompt = f"以下の資料を参考に、ユーザーの質問に正確に回答してください。\n資料にないことは「わかりません」と答えてください。\n\n【資料】\n{context}\n【質問】\n{query}"
    return prompt
