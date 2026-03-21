import socket
import pickle
import numpy as np
import time
import argparse

QUERY_FILES = {
    "fresh": "data/queries_fresh.npy",
    "partial": "data/queries_partial.npy",
    "stale": "data/queries_stale.npy",
}

GT_FILE = "data/gt_top100.npy"

HOST = "127.0.0.1"
PORT = 6000  # proxy


def send_query(q):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))

    sock.sendall(pickle.dumps(q))

    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk

    sock.close()
    return pickle.loads(data)


def recall_at_k(pred, gt, k):
    total = 0
    for i in range(len(pred)):
        total += len(set(pred[i][:k]) & set(gt[i][:k])) / k
    return total / len(pred)


def run():
    gt = np.load(GT_FILE)

    for freshness, file in QUERY_FILES.items():
        queries = np.load(file)

        results = []
        latencies = []

        for q in queries:
            t0 = time.time()

            _, I = send_query(q)

            t1 = time.time()

            latencies.append((t1 - t0) * 1000)
            results.append(I[0])

        results = np.array(results)

        r10 = recall_at_k(results, gt, 10)

        print(f"{freshness} | Recall@10={r10:.4f} | P95 latency={np.percentile(latencies,95):.2f} ms")


if __name__ == "__main__":
    run()