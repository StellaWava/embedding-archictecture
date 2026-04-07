"""
Step 11: 
This is architectural behaviour experiment

"""
import faiss
import numpy as np
from tqdm import tqdm

# =========================
# Paths
# =========================
HNSW_INDEX = "data/hnsw.index"
IVF_INDEX = "data/ivf_flat.index"

Q_FRESH = "data/queries_fresh.npy"
Q_PARTIAL = "data/queries_patial.npy"
Q_STALE = "data/queries_stale.npy"

GT_FILE = "data/gt_top100.npy"

TOPK = 100

# =========================
# Load data
# =========================
print("Loading indexes and data...")

hnsw = faiss.read_index(HNSW_INDEX)
ivf = faiss.read_index(IVF_INDEX)

gt = np.load(GT_FILE)

query_sets = {
    "fresh": np.load(Q_FRESH),
    "partial": np.load(Q_PARTIAL),
    "stale": np.load(Q_STALE),
}

# =========================
# Recall computation
# =========================
def compute_recall(I, gt, k=TOPK):
    hits = 0
    total = I.shape[0] * k

    for i in range(I.shape[0]):
        hits += len(set(I[i][:k]) & set(gt[i][:k]))

    return hits / total


# =========================
# HNSW evaluation
# =========================
def eval_hnsw(index, queries, ef_values):
    results = []

    for ef in ef_values:
        index.hnsw.efSearch = ef

        D, I = index.search(queries, TOPK)
        recall = compute_recall(I, gt)

        results.append({
            "efSearch": ef,
            "recall": recall,
        })

    return results


# =========================
# IVF evaluation
# =========================
def eval_ivf(index, queries, nprobe_values):
    results = []

    for nprobe in nprobe_values:
        index.nprobe = nprobe

        D, I = index.search(queries, TOPK)
        recall = compute_recall(I, gt)

        results.append({
            "nprobe": nprobe,
            "recall": recall,
        })

    return results


# =========================
# IVF routing accuracy
# =========================
def compute_centroid_hit_rate(index, queries, gt):
    print("Computing centroid routing accuracy...")

    quantizer = index.quantizer

    # assign queries to centroids
    _, assigned = quantizer.search(queries, 1)

    hits = 0

    # reconstruct all vectors once (efficient enough for your scale)
    all_vecs = index.reconstruct_n(0, index.ntotal)

    for i in tqdm(range(len(queries))):
        gt_ids = gt[i]
        gt_vecs = all_vecs[gt_ids]

        _, gt_centroids = quantizer.search(gt_vecs, 1)

        if assigned[i][0] in gt_centroids:
            hits += 1

    return hits / len(queries)


# =========================
# Run experiment
# =========================
# def run():
#     ef_values = [16, 32, 64, 128, 256]
#     nprobe_values = [1, 4, 8, 16, 32]

#     for name, queries in query_sets.items():
#         print("\n==============================")
#         print(f"DATASET: {name.upper()}")
#         print("==============================")

#         # ---- HNSW ----
#         print("\nHNSW results:")
#         hnsw_res = eval_hnsw(hnsw, queries, ef_values)
#         for r in hnsw_res:
#             print(r)

#         # ---- IVF ----
#         print("\nIVF results:")
#         ivf_res = eval_ivf(ivf, queries, nprobe_values)
#         for r in ivf_res:
#             print(r)

#         # ---- IVF routing ----
#         hit_rate = compute_centroid_hit_rate(ivf, queries, gt)
#         print(f"\nIVF centroid hit rate: {hit_rate:.4f}")
import csv
import os
OUTPUT_FILE = "results/mechanism_results.csv"
def run():
    ef_values = [16, 32, 64, 128, 256]
    nprobe_values = [1, 4, 8, 16, 32]

    os.makedirs("results", exist_ok=True)
    write_header = not os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "arch",
                "freshness",
                "knob_name",
                "knob_value",
                "recall",
                "centroid_hit_rate",  # only used for IVF
            ])

        for name, queries in query_sets.items():
            print("\n==============================")
            print(f"DATASET: {name.upper()}")
            print("==============================")

            # -----------------
            # HNSW
            # -----------------
            for ef in ef_values:
                hnsw.hnsw.efSearch = ef

                D, I = hnsw.search(queries, TOPK)
                recall = compute_recall(I, gt)

                row = [
                    "hnsw",
                    name,
                    "efSearch",
                    ef,
                    round(recall, 6),
                    "",  # no centroid metric
                ]

                writer.writerow(row)
                print(row)

            # -----------------
            # IVF
            # -----------------
            # compute centroid hit rate once per dataset
            hit_rate = compute_centroid_hit_rate(ivf, queries, gt)

            for nprobe in nprobe_values:
                ivf.nprobe = nprobe

                D, I = ivf.search(queries, TOPK)
                recall = compute_recall(I, gt)

                row = [
                    "ivf",
                    name,
                    "nprobe",
                    nprobe,
                    round(recall, 6),
                    round(hit_rate, 6),
                ]

                writer.writerow(row)
                print(row)

if __name__ == "__main__":
    run()