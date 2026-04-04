#setting the ground truth similarity search | fresh top-k, no network injections
import os

import numpy as np
from tqdm import tqdm

CORPUS_FILE = "data/corpus.npy"
QUERIES_FILE = "data/queries_fresh.npy"
OUTPUT_FILE = "data/gt_top100.npy"

TOPK = 100
CHUNK_SIZE = 256

os.makedirs("data", exist_ok=True)

#load data 
corpus = np.load(CORPUS_FILE).astype("float32")
queries = np.load(QUERIES_FILE).astype("float32")

assert corpus.ndim == 2
assert queries.ndim == 2
assert corpus.shape[1] == queries.shape[1]

print("Corpus shape:", corpus.shape)
print("Queries shape:", queries.shape)

# cosine similarity because vectors are normalized
gt_indices = []

for start in tqdm(range(0, queries.shape[0], CHUNK_SIZE), desc="Ground truth"):
    q = queries[start:start + CHUNK_SIZE]  # [b, d]
    sims = q @ corpus.T                    # [b, n]
    top_idx = np.argpartition(-sims, kth=TOPK - 1, axis=1)[:, :TOPK]

    # sort the top-k exactly
    row_sorted = np.take_along_axis(
        top_idx,
        np.argsort(-np.take_along_axis(sims, top_idx, axis=1), axis=1),
        axis=1,
    )
    gt_indices.append(row_sorted.astype("int32"))

gt_indices = np.vstack(gt_indices)
np.save(OUTPUT_FILE, gt_indices)

print(f"Saved exact top-{TOPK} ground truth to {OUTPUT_FILE} with shape {gt_indices.shape}")
