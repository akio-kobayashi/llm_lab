import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# --- 編集可能: モデル設定 ---
DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# --- 編集可能ここまで ---

class FaissRAGPipeline:
    """
    FAISSを利用したRAG（Retrieval-Augmented Generation）パイプラインを管理するクラス。
    """
    def __init__(self, embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID):
        """
        コンストラクタ。Embeddingモデルをロードする。
        """
        print(f"Loading embedding model: {embedding_model_id}")
        self.embedding_model = SentenceTransformer(embedding_model_id, trust_remote_code=True)
        self.index = None
        self.documents = []
        print("Embedding model loaded.")

    def build_index(self, docs_path: str):
        """
        JSONL形式のドキュメントファイルからFAISSインデックスを構築する。

        Args:
            docs_path (str): ドキュメントファイルへのパス (JSONL形式)。
        """
        print(f"Building FAISS index from: {docs_path}")
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Document file not found: {docs_path}")

        # ドキュメントの読み込み
        self.documents = []
        with open(docs_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.documents.append(json.loads(line))

        # テキストチャンクのリストを作成
        texts = [doc['text'] for doc in self.documents]

        # Embeddingの計算 (プログレスバー付き)
        print("Encoding documents...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # L2正規化 (内積でコサイン類似度を計算するため)
        faiss.normalize_L2(embeddings)
        
        d = embeddings.shape[1]  # ベクトルの次元
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        
        print(f"Index built successfully. Total documents: {self.index.ntotal}")

    def save_index(self, index_path: str, docs_meta_path: str):
        """
        FAISSインデックスとドキュメントメタデータを保存する。

        Args:
            index_path (str): FAISSインデックスの保存先パス。
            docs_meta_path (str): ドキュメントメタデータ (JSON) の保存先パス。
        """
        if self.index is None:
            raise ValueError("Index is not built yet. Call build_index() first.")
        
        print(f"Saving FAISS index to: {index_path}")
        faiss.write_index(self.index, index_path)
        
        print(f"Saving documents metadata to: {docs_meta_path}")
        with open(docs_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load_index(self, index_path: str, docs_meta_path: str):
        """
        FAISSインデックスとドキュメントメタデータを読み込む。

        Args:
            index_path (str): FAISSインデックスのパス。
            docs_meta_path (str): ドキュメントメタデータ (JSON) のパス。
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(docs_meta_path):
            raise FileNotFoundError(f"Documents metadata file not found: {docs_meta_path}")

        print(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading documents metadata from: {docs_meta_path}")
        with open(docs_meta_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
            
        print(f"Index and metadata loaded. Total documents: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 3):
        """
        クエリに類似したドキュメントを検索する。

        Args:
            query (str): 検索クエリ。
            top_k (int): 取得する上位ドキュメント数。

        Returns:
            list: 類似ドキュメントのリスト (dict形式)。
        """
        if self.index is None:
            raise ValueError("Index is not loaded. Call load_index() or build_index() first.")
        
        print(f"Searching for: '{query}' (top_k={top_k})")
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # L2正規化
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(top_k):
            doc_index = indices[0][i]
            if doc_index < len(self.documents):
                results.append(self.documents[doc_index])
        
        print(f"Found {len(results)} relevant documents.")
        return results

    def create_prompt_with_context(self, query: str, context_docs: list):
        """
        検索結果を埋め込んだプロンプトを作成する。

        Args:
            query (str): 元の質問。
            context_docs (list): 検索されたドキュメントのリスト。

        Returns:
            str: 生成されたプロンプト。
        """
        context = "\n\n---\n\n".join([f"■参考情報:\n{doc['text']}" for doc in context_docs])
        
        prompt = f"""以下の参考情報に基づいて、質問に答えてください。

{context}

---
質問: {query}
回答:"""
        return prompt

if __name__ == '__main__':
    # このファイルが直接実行された場合の動作テスト
    print("--- Running rag.py self-test ---")

    # ダミーデータとパスの準備
    DUMMY_DOCS_PATH = "dummy_docs.jsonl"
    DUMMY_INDEX_PATH = "dummy.index"
    DUMMY_META_PATH = "dummy_docs.json"

    dummy_data = [
        {"doc_id": "test01", "text": "日本の首都は東京です。"},
        {"doc_id": "test02", "text": "世界で一番高い山はエベレストです。"},
        {"doc_id": "test03", "text": "フランスの首都はパリです。"},
    ]
    with open(DUMMY_DOCS_PATH, 'w', encoding='utf-8') as f:
        for item in dummy_data:
            f.write(json.dumps(item) + '\n')

    try:
        # 1. パイプラインの初期化
        rag_pipeline = FaissRAGPipeline()

        # 2. インデックスの構築と保存
        rag_pipeline.build_index(DUMMY_DOCS_PATH)
        rag_pipeline.save_index(DUMMY_INDEX_PATH, DUMMY_META_PATH)

        # 3. (別インスタンスで) インデックスの読み込み
        new_rag_pipeline = FaissRAGPipeline()
        new_rag_pipeline.load_index(DUMMY_INDEX_PATH, DUMMY_META_PATH)

        # 4. 検索の実行
        query = "日本の首都はどこですか？"
        results = new_rag_pipeline.search(query, top_k=1)
        
        print("\n--- Search Results ---")
        for res in results:
            print(res)
        assert len(results) == 1
        assert results[0]['doc_id'] == 'test01'

        # 5. プロンプトの作成
        prompt = new_rag_pipeline.create_prompt_with_context(query, results)
        print("\n--- Generated Prompt ---")
        print(prompt)
        assert "東京" in prompt

        print("\n--- Self-test finished successfully ---")

    except Exception as e:
        print(f"\nSelf-test failed: {e}")
    finally:
        # クリーンアップ
        if os.path.exists(DUMMY_DOCS_PATH):
            os.remove(DUMMY_DOCS_PATH)
        if os.path.exists(DUMMY_INDEX_PATH):
            os.remove(DUMMY_INDEX_PATH)
        if os.path.exists(DUMMY_META_PATH):
            os.remove(DUMMY_META_PATH)
