"""

STEP 6: (Local experiment execute)

It evaluates:
architecture: hnsw vs ivf
freshness: fresh, partial, stale
effort knob:
HNSW: efSearch
IVF: nprobe

It reports recall and latency.

"""

import argparse
import csv
import os
import time

import faiss
import numpy as np

CORPUS_FILE = "data/corpus.npy"
GT_FILE = "data/gt_top100.npy"

#note fresh is same as partial, same as stale in this setting. 
QUERY_FILES = {
    "fresh": "data/queries_fresh.npy",
    "partial": "data/queries_patial.npy",
    "stale": "data/queries_stale.npy",
}

INDEX_FILES = {
    "hnsw": "data/hnsw.index",
    "ivf": "data/ivf_flat.index",
}


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    pred_k = pred[:, :k]
    gt_k = gt[:, :k]

    total = 0.0
    for i in range(pred_k.shape[0]):
        total += len(set(pred_k[i].tolist()) & set(gt_k[i].tolist())) / float(k)
    return total / pred_k.shape[0]

#introduce dynamic search evolution
def alpha_from_time(dt, k=0.05):
    """
    Sublinear drift magnitude as a function of time.
    dt: shape (B,)
    """
    return k * np.sqrt(dt)   # sublinear

import numpy as np
import time


def alpha_from_time(dt, k=0.5):
    """
    Sublinear drift magnitude as a function of time.
    dt: shape (B,)
    """
    return k * np.sqrt(dt)


def batched_search(index, queries, topk, batch_size, steps=3):
    all_I = []
    latencies_ms = []

    for start in range(0, queries.shape[0], batch_size):
        q = queries[start:start + batch_size]
        current_q = q.copy()

        B, D = current_q.shape

        # per-query accumulated time
        cumulative_dt = np.zeros(B, dtype="float32")

        # track previous alpha for incremental drift
        prev_alpha = np.zeros(B, dtype="float32")

        for step in range(steps):
            # --- ANN search ---
            t0 = time.perf_counter()
            _, I = index.search(current_q, topk)
            t1 = time.perf_counter()

            batch_ms = (t1 - t0) * 1000.0
            per_query_ms = batch_ms / B

            # --- accumulate time exposure ---
            cumulative_dt += per_query_ms

            # --- compute sublinear drift ---
            alpha = alpha_from_time(cumulative_dt)        # shape (B,)
            delta_alpha = alpha - prev_alpha              # incremental drift

            # --- generate normalized noise ---
            noise = np.random.randn(B, D).astype("float32")
            noise /= np.linalg.norm(noise, axis=1, keepdims=True)

            # --- apply incremental drift ---
            current_q = current_q + delta_alpha[:, None] * noise

            # re-normalize to stay on unit sphere
            current_q /= np.linalg.norm(current_q, axis=1, keepdims=True)

            prev_alpha = alpha

        all_I.append(I)

        # final per-query "time exposure" (your effective delay)
        latencies_ms.extend(cumulative_dt.tolist())

    return np.vstack(all_I), np.array(latencies_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["hnsw", "ivf"], required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="results/local_resultsdt.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    gt = np.load(GT_FILE).astype("int32")
    index = faiss.read_index(INDEX_FILES[args.arch])

    if args.arch == "hnsw":
        knob_name = "efSearch"
        knob_values = [16, 32, 64, 128, 256]
    else:
        knob_name = "nprobe"
        knob_values = [1, 2, 4, 8, 16, 32]

    write_header = not os.path.exists(args.output)

    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "arch",
                "freshness",
                "knob_name",
                "knob_value",
                "topk",
                "num_queries",
                "recall_at_10",
                "recall_at_100",
                "latency_p50_ms",
                "latency_p95_ms",
                "latency_p99_ms",
                "qps",
            ])

        for freshness, query_file in QUERY_FILES.items():
            queries = np.load(query_file).astype("float32")

            for knob in knob_values:
                if args.arch == "hnsw":
                    index.hnsw.efSearch = knob
                else:
                    index.nprobe = knob

                I, lat_ms = batched_search(
                    index=index,
                    queries=queries,
                    topk=max(args.topk, 100),
                    batch_size=args.batch_size,
                )

                r10 = recall_at_k(I, gt, 10)
                r100 = recall_at_k(I, gt, 100)

                total_time_s = lat_ms.sum() / 1000.0
                qps = len(lat_ms) / total_time_s if total_time_s > 0 else 0.0

                row = [
                    args.arch,
                    freshness,
                    knob_name,
                    knob,
                    args.topk,
                    queries.shape[0],
                    round(r10, 6),
                    round(r100, 6),
                    round(float(np.percentile(lat_ms, 50)), 4),
                    round(float(np.percentile(lat_ms, 95)), 4),
                    round(float(np.percentile(lat_ms, 99)), 4),
                    round(float(qps), 4),
                ]
                writer.writerow(row)
                print(row)


if __name__ == "__main__":
    main() 