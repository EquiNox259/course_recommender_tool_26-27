import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the verified paths
FAISS_INDEX_PATH = r"./data/course_index_new.faiss"
COURSE_META_PATH = r"./data/courses_metadata_new.csv"

index = faiss.read_index(FAISS_INDEX_PATH)
df = pd.read_csv(COURSE_META_PATH)

# Load the exact same embedding model your project config uses
# (If your project uses a different path, replace this string)
print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2") 

print("\n=== RUNNING SEMANTIC REVERSE-ENGINEERING TEST ===")

# Pick a mid-sheet course row that definitely has a full description text
sample_row = df.dropna(subset=['Description', 'Course Name']).iloc[len(df) // 2]
print(f"Targeting Course for Verification: {sample_row['Course Code']} - {sample_row['Course Name']}")

# Define the three distinct possibilities of what was embedded
test_cases = {
    "ONLY Course Name": str(sample_row['Course Name']),
    "ONLY Description": str(sample_row['Description']),
    "COMBINED Name + Description": f"{sample_row['Course Name']} {sample_row['Description']}"
}

for label, text_content in test_cases.items():
    # 1. Turn the test text into a math vector
    query_emb = model.encode(text_content, normalize_embeddings=True).astype(np.float32)
    
    # 2. Query the FAISS file to find the single closest match
    scores, indices = index.search(query_emb.reshape(1, -1), k=1)
    
    matched_idx = indices[0][0]
    matched_score = scores[0][0]
    matched_code = df.iloc[matched_idx]['Course Code']
    
    print(f"\nTesting if index was built using [{label}]:")
    print(f"   Top Matched Row in DB: {matched_code}")
    print(f"   Similarity Score:      {matched_score:.4f}")
    
    # A score close to 1.0000 means this exact text sequence is what lives in the file!
    if matched_code == sample_row['Course Code'] and matched_score > 0.98:
        print(f"   🏆 FOUND A MATCH! The FAISS file was baked using {label}!")