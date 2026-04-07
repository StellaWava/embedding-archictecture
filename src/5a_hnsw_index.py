"""
STEP 5 a) (Local Experiment)

Search Index - hnsw
"""


import os

import faiss
import numpy as np

CORPUS_FILE = "data/corpus.npy"
INDEX_FILE = "data/hnsw.index"

M = 32
EF_CONSTRUCTION = 200

os.makedirs("data", exist_ok=True)

xb = np.load(CORPUS_FILE).astype("float32")
d = xb.shape[1]

print("Building HNSW index...")
index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = EF_CONSTRUCTION

index.add(xb)

faiss.write_index(index, INDEX_FILE)
print(f"Saved HNSW index to {INDEX_FILE}")
print("ntotal =", index.ntotal)