"""
STEP 3: (Local experiment)

Creating query conditions for the empirical drift checks

"""

import json
import os 

import numpy as np 

#base locality 
INPUT_QUERIES = "data/metadata_queries.npy"
OUT_FRESH = "data/queries_fresh.npy"
OUT_PARTIAL = "data/queries_patial.npy"
OUT_STALE = "data/queries_stale.npy"
METAL_FILE = "data/metadata_drift.json"


#set 
SEED = 42

#drift magnitude in embedding space 
# PARTIAL_ALPHA = 0.15
# STALE_ALPHA = 0.35 
PARTIAL_ALPHA = 0
STALE_ALPHA = 0

np.random.seed(SEED)
os.makedirs("data", exist_ok=True)

#normalize queries 
def l2_normalize(x: np.array) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x/norms

def make_drift_queries(q: np.ndarray, alpha: float) -> np.ndarray:
    noise = np.random.randn(*q.shape).astype("float32")
    noise = l2_normalize(noise)

    drifted = q + alpha*noise  #changed from + for *
    drifted = l2_normalize(drifted.astype("float32"))
    return drifted.astype("float32")

queries = np.load(INPUT_QUERIES).astype("float32")
queries = l2_normalize(queries)

queries_fresh = queries.copy()
queries_partial = make_drift_queries(queries, PARTIAL_ALPHA)
queries_stale = make_drift_queries(queries, STALE_ALPHA)


#save generated queries
np.save(OUT_FRESH, queries_fresh)
np.save(OUT_PARTIAL, queries_partial)
np.save(OUT_STALE, queries_stale)

with open(METAL_FILE, "w") as f:
    json.dump(
        {
            "partial_alpha": PARTIAL_ALPHA,
            "stale_alpha": STALE_ALPHA,
            "num_queries": int(queries.shape[0]),
            "embedding_dim": int(queries.shape[1]),
            "note": "fresh is treated as current true query embedding; partial/stale simulate missed background updates"
        },
        f,
        indent=2,
    )

print("Saved:")
print(" ", OUT_FRESH, queries_fresh.shape)
print(" ", OUT_PARTIAL, queries_partial.shape)
print(" ", OUT_STALE, queries_stale.shape)