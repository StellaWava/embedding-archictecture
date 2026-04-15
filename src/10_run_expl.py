"""
STEP 10: Upgrading FAISS calls with RPC calls
for latency in 5 20 50 100; do   for jitter in 0 5 10; do     for loss in 0.0 0.01 0.05; do       python3 src/10_run_expl.py         --arch hnsw         --latency $latency         --jitter $jitter         --loss $loss         --bandwidth 100;     done;   done; done

"""


# import socket
# import pickle
# import numpy as np
# import time
# import csv
# import os
# import argparse

# QUERY_FILES = {
#     "fresh": "data/queries_fresh.npy",
#     "partial": "data/queries_patial.npy",
#     "stale": "data/queries_stale.npy",
# }

# GT_FILE = "data/gt_top100.npy"

# HOST = "127.0.0.1"
# PORT = 6000

# BUFFER_SIZE = 4096
# TIMEOUT = 5


# def recv_until_eof(sock):
#     data = b""
#     while True:
#         chunk = sock.recv(BUFFER_SIZE)
#         if not chunk:
#             break
#         data += chunk
#     return data


# def send_query(q):
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.settimeout(TIMEOUT)

#     try:
#         sock.connect((HOST, PORT))

#         payload = pickle.dumps(q)
#         sock.sendall(payload)

#         # Critical: signal end of request
#         sock.shutdown(socket.SHUT_WR)

#         response = recv_until_eof(sock)

#         return pickle.loads(response), False  # success

#     except Exception:
#         return None, True  # failure

#     finally:
#         sock.close()


# def recall_at_k(pred, gt, k):
#     total = 0.0
#     for i in range(len(pred)):
#         total += len(set(pred[i][:k]) & set(gt[i][:k])) / k
#     return total / len(pred)


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--latency", type=float, required=True)
#     parser.add_argument("--jitter", type=float, required=True)
#     parser.add_argument("--loss", type=float, required=True)
#     parser.add_argument("--bandwidth", type=float, required=True)

#     parser.add_argument("--arch", type=str, required=True)
#     parser.add_argument("--output", default="results/network_results.csv")

#     args = parser.parse_args()

#     os.makedirs("results", exist_ok=True)

#     gt = np.load(GT_FILE)

#     write_header = not os.path.exists(args.output)

#     with open(args.output, "a", newline="") as f:
#         writer = csv.writer(f)

#         if write_header:
#             writer.writerow([
#                 "arch",
#                 "freshness",
#                 "latency_ms",
#                 "jitter_ms",
#                 "packet_loss",
#                 "bandwidth_mbps",
#                 "num_queries",
#                 "failures",
#                 "recall_at_10",
#                 "latency_p50_ms",
#                 "latency_p95_ms",
#                 "latency_p99_ms",
#             ])

#         for freshness, file in QUERY_FILES.items():

#             queries = np.load(file)

#             results = []
#             latencies = []
#             failures = 0
#             valid_gt = []

#             for i, q in enumerate(queries):

#                 t0 = time.time()
#                 response, failed = send_query(q)
#                 t1 = time.time()

#                 if failed:
#                     failures += 1
#                     continue

#                 _, I = response

#                 latencies.append((t1 - t0) * 1000.0)
#                 results.append(I[0])
#                 valid_gt.append(gt[i])  # keep alignment

#             if len(results) == 0:
#                 r10 = 0.0
#                 p50 = p95 = p99 = 0.0
#             else:
#                 results = np.array(results)
#                 valid_gt = np.array(valid_gt)

#                 r10 = recall_at_k(results, valid_gt, 10)

#                 p50 = np.percentile(latencies, 50)
#                 p95 = np.percentile(latencies, 95)
#                 p99 = np.percentile(latencies, 99)

#             row = [
#                 args.arch,
#                 freshness,
#                 args.latency,
#                 args.jitter,
#                 args.loss,
#                 args.bandwidth,
#                 len(queries),
#                 failures,
#                 round(r10, 6),
#                 round(p50, 4),
#                 round(p95, 4),
#                 round(p99, 4),
#             ]

#             writer.writerow(row)
#             print(row)


# if __name__ == "__main__":
#     main() 


import numpy as np
import socket
import pickle
import time
import argparse
import csv
import os

import argparse
import csv
import os
import pickle
import socket
import time

import numpy as np

GT_FILE = "data/gt_top100.npy"
#note fresh = partial = stale = same query
QUERY_FILES = {
    "fresh": "data/queries_fresh.npy",
    "partial": "data/queries_patial.npy",
    "stale": "data/queries_stale.npy",
}

HOST = "127.0.0.1"
PORT = 5000
BUFFER_SIZE = 4096
TIMEOUT = 5

KNOB_VALUES = {
    "hnsw": [16, 32, 64, 128, 256],
    "ivf":  [1, 2, 4, 8, 16, 32],
}
KNOB_NAME = {
    "hnsw": "efSearch",
    "ivf":  "nprobe",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def alpha_from_time(dt, k=0.5):
    """Sublinear drift magnitude. dt: scalar or array of ms."""
    return k * np.sqrt(dt)


def recv_until_eof(sock):
    data = b""
    while True:
        chunk = sock.recv(BUFFER_SIZE)
        if not chunk:
            break
        data += chunk
    return data


def send_query(q, net_params, arch, knob):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)
    try:
        sock.connect((HOST, PORT))
        payload = pickle.dumps({
            "query":     q.astype("float32"),
            "latency":   net_params["latency"],
            "jitter":    net_params["jitter"],
            "bandwidth": net_params["bandwidth"],
            "arch":      arch,
            "knob":      knob,          # FIX 3: knob forwarded to server
        })
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        response = recv_until_eof(sock)
        return pickle.loads(response), False
    except Exception:
        return None, True
    finally:
        sock.close()


def recall_at_k(pred, gt, k):
    total = 0.0
    for i in range(len(pred)):
        total += len(set(pred[i][:k]) & set(gt[i][:k])) / k
    return total / len(pred)


# ── core search loop with drift ───────────────────────────────────────────────

def batched_search_network(queries, net_params, arch, knob,
                           topk=100, batch_size=64, steps=3, packet_loss=0.0):
    """
    Mirrors local batched_search but drives drift from real observed
    network round-trip time instead of index search time.
    """
    all_I      = []
    latencies  = []
    failures   = 0
    valid_idx  = []

    D = queries.shape[1]

    for start in range(0, queries.shape[0], batch_size):
        batch_raw = queries[start : start + batch_size]
        B = batch_raw.shape[0]

        current_q    = batch_raw.copy()
        cumulative_dt = np.zeros(B, dtype="float32")
        prev_alpha    = np.zeros(B, dtype="float32")

        batch_I   = [None] * B
        batch_lat = [None] * B
        dropped   = np.zeros(B, dtype=bool)

        for step in range(steps):
            for b in range(B):
                if dropped[b]:
                    continue

                # FIX 1 (client-side jitter): add gaussian jitter to base latency
                # so each query in the batch sees independent delay variation
                jitter_sample = np.random.normal(0, net_params["jitter"])  # FIX 2
                effective_latency = max(0.0, net_params["latency"] + jitter_sample)
                

                # packet loss
                if np.random.rand() < packet_loss:
                    failures += 1
                    dropped[b] = True
                    continue

                t0 = time.time()
                time.sleep(effective_latency / 1000.0)
                
                response, failed = send_query(
                    current_q[b : b + 1], net_params, arch, knob
                )
                t1 = time.time()

                if failed or response is None:
                    failures += 1
                    dropped[b] = True
                    continue

                _, I = response
                observed_ms = (t1 - t0) * 1000.0

                # FIX 1: accumulate *observed* round-trip time as the drift clock
                cumulative_dt[b] += observed_ms

                # compute incremental drift from observed latency
                alpha      = alpha_from_time(cumulative_dt[b])
                delta_a    = alpha - prev_alpha[b]
                prev_alpha[b] = alpha

                noise = np.random.randn(D).astype("float32")
                noise /= np.linalg.norm(noise)
                current_q[b] = current_q[b] + delta_a * noise
                current_q[b] /= np.linalg.norm(current_q[b])

                # keep last step's results
                batch_I[b]   = I[0]
                batch_lat[b] = cumulative_dt[b]

        for b in range(B):
            if not dropped[b] and batch_I[b] is not None:
                all_I.append(batch_I[b])
                latencies.append(batch_lat[b])
                valid_idx.append(start + b)

    return np.array(all_I), np.array(latencies), failures, valid_idx


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",       choices=["hnsw", "ivf"], required=True)
    parser.add_argument("--latency",    type=float, required=True)
    parser.add_argument("--jitter",     type=float, required=True)
    parser.add_argument("--loss",       type=float, required=True)
    parser.add_argument("--bandwidth",  type=float, required=True)
    parser.add_argument("--topk",       type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--steps",      type=int,   default=3)
    parser.add_argument("--output",     type=str,   default="results/network_results.csv")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    gt          = np.load(GT_FILE)
    net_params  = {
        "latency":   args.latency,
        "jitter":    args.jitter,
        "bandwidth": args.bandwidth,
    }
    knob_name   = KNOB_NAME[args.arch]

    write_header = not os.path.exists(args.output)

    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "arch", "freshness", "knob_name", "knob_value",
                "latency_ms", "jitter_ms", "packet_loss", "bandwidth_mbps",
                "num_queries", "failures",
                "recall_at_10",
                "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                "qps",
            ])

        for freshness, query_file in QUERY_FILES.items():
            queries = np.load(query_file).astype("float32")

            # FIX 3: sweep knob values just like the local benchmark
            for knob in KNOB_VALUES[args.arch]:

                I, lat_ms, failures, valid_idx = batched_search_network(
                    queries      = queries,
                    net_params   = net_params,
                    arch         = args.arch,
                    knob         = knob,
                    topk         = max(args.topk, 100),
                    batch_size   = args.batch_size,
                    steps        = args.steps,
                    packet_loss  = args.loss,
                )

                if len(I) == 0:
                    r10 = p50 = p95 = p99 = qps = 0.0
                else:
                    valid_gt  = gt[valid_idx]
                    r10       = recall_at_k(I, valid_gt, 10)
                    total_s   = lat_ms.sum() / 1000.0
                    qps       = len(lat_ms) / total_s if total_s > 0 else 0.0
                    p50, p95, p99 = (
                        np.percentile(lat_ms, 50),
                        np.percentile(lat_ms, 95),
                        np.percentile(lat_ms, 99),
                    )

                row = [
                    args.arch, freshness, knob_name, knob,
                    args.latency, args.jitter, args.loss, args.bandwidth,
                    len(queries), failures,
                    round(r10,  6),
                    round(float(p50), 4),
                    round(float(p95), 4),
                    round(float(p99), 4),
                    round(float(qps), 4),
                ]
                writer.writerow(row)
                print(row)


if __name__ == "__main__":
    main()