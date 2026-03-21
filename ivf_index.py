import os

import faiss
import numpy as np

CORPUS_FILE = "data/corpus.npy"
INDEX_FILE = "data/ivf_flat.index"

N_LIST = 256
TRAIN_SIZE = 50000
SEED = 42

os.makedirs("data", exist_ok=True)
np.random.seed(SEED)

xb = np.load(CORPUS_FILE).astype("float32")
d = xb.shape[1]

if xb.shape[0] < TRAIN_SIZE:
    train_x = xb
else:
    idx = np.random.choice(xb.shape[0], size=TRAIN_SIZE, replace=False)
    train_x = xb[idx]

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, N_LIST, faiss.METRIC_INNER_PRODUCT)

print("Training IVF index...")
index.train(train_x)

print("Adding corpus vectors...")
index.add(xb)

faiss.write_index(index, INDEX_FILE)
print(f"Saved IVF index to {INDEX_FILE}")
print("ntotal =", index.ntotal)