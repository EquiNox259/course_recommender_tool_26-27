import faiss
import numpy as np
class SemanticCache:
    def __init__(self,
                 embedding_dim=384,
                 threshold=0.90):
        self.threshold = threshold
        self.embedding_dim = embedding_dim
        # cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        # cache_id -> response
        self.responses = {}
        # faiss row -> cache_id
        self.id_map = []
        self.next_id = 0

    def lookup(self, embedding):
        if self.index.ntotal == 0:
            return None
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)
        faiss.normalize_L2(embedding)
        scores, indices = self.index.search(embedding, 1)
        similarity = scores[0][0]
        idx = indices[0][0]
        if idx == -1:
            return None
        if similarity < self.threshold:
            return None
        cache_id = self.id_map[idx]
        print(f"[CACHE HIT] similarity={similarity:.3f}")
        return self.responses[cache_id]

    def add(self,
            query,
            embedding,
            response):
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        cache_id = self.next_id
        self.next_id += 1
        self.id_map.append(cache_id)
        self.responses[cache_id] = {
            "query": query,
            "response": response
        }