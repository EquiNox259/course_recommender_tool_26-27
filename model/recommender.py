from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os
from config import MODEL_ASSETS_DIR, MODEL_DATA_DIR

MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, "course_encoder_new")
FAISS_INDEX_PATH = os.path.join(MODEL_DATA_DIR, "course_index_new.faiss")
COURSE_META_PATH = os.path.join(MODEL_DATA_DIR, "courses_metadata_new.csv")

model = SentenceTransformer(MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)
df = pd.read_csv(COURSE_META_PATH)


def search_courses(query, top_k=10):
    """
    Search for the most relevant courses given a text query.

    Parameters
    ----------
    query : str
        Natural language query (e.g. "markov chain")
    top_k : int
        Number of results to return

    Returns
    -------
    pandas.DataFrame
        Top-k matching courses with similarity scores
    """

    # Encode query
    query_emb = model.encode(
        query,
        normalize_embeddings=True
    ).astype(np.float32)

    # FAISS expects shape (n_queries, dim)
    scores, indices = index.search(query_emb.reshape(1, -1), top_k)

    # Fetch results
    results = df.iloc[indices[0]].copy()
    results["score"] = scores[0]

    return results[["Course Code", "Course Name", "score"]]

'''res = search_courses("markov chain")
print(res)'''

def get_candidate_courses(query, top_k=10):
    """
    Returns list of dicts:
    [{code: ..., name: ...}, ...]
    """
    results = search_courses(query, top_k)

    return [
        {
            "code": row["Course Code"],
            "name": row["Course Name"]
        }
        for _, row in results.iterrows()
    ]

