# import socket
# import pickle
# import numpy as np
# import time

# QUERY_FILES = {
#     "fresh": "data/queries_fresh.npy",
#     "partial": "data/queries_patial.npy",
#     "stale": "data/queries_stale.npy",
# }

# GT_FILE = "data/gt_top100.npy"

# HOST = "127.0.0.1"
# PORT = 6000

# BUFFER_SIZE = 4096


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
#     try:
#         sock.connect((HOST, PORT))

#         payload = pickle.dumps(q)
#         sock.sendall(payload)

#         # Important: signal that request is complete
#         sock.shutdown(socket.SHUT_WR)

#         response = recv_until_eof(sock)
#         return pickle.loads(response)
#     finally:
#         sock.close()


# def recall_at_k(pred, gt, k):
#     total = 0.0
#     for i in range(len(pred)):
#         total += len(set(pred[i][:k]) & set(gt[i][:k])) / k
#     return total / len(pred)


# def run():
#     gt = np.load(GT_FILE)

#     for freshness, file in QUERY_FILES.items():
#         queries = np.load(file)

#         results = []
#         latencies = []

#         for q in queries:
#             t0 = time.time()
#             _, I = send_query(q)
#             t1 = time.time()

#             latencies.append((t1 - t0) * 1000.0)
#             results.append(I[0])

#         results = np.array(results)

#         r10 = recall_at_k(results, gt, 10)
#         p95 = np.percentile(latencies, 95)

#         print(f"{freshness} | Recall@10={r10:.4f} | P95 latency={p95:.2f} ms")


# if __name__ == "__main__":
#     run()


import socket
import pickle
import numpy as np
import time
import csv
import os
import argparse

QUERY_FILES = {
    "fresh": "data/queries_fresh.npy",
    "partial": "data/queries_patial.npy",
    "stale": "data/queries_stale.npy",
}

GT_FILE = "data/gt_top100.npy"

HOST = "127.0.0.1"
PORT = 6000

BUFFER_SIZE = 4096
TIMEOUT = 5


def recv_until_eof(sock):
    data = b""
    while True:
        chunk = sock.recv(BUFFER_SIZE)
        if not chunk:
            break
        data += chunk
    return data


def send_query(q):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)

    try:
        sock.connect((HOST, PORT))

        payload = pickle.dumps(q)
        sock.sendall(payload)

        # Critical: signal end of request
        sock.shutdown(socket.SHUT_WR)

        response = recv_until_eof(sock)

        return pickle.loads(response), False  # success

    except Exception:
        return None, True  # failure

    finally:
        sock.close()


def recall_at_k(pred, gt, k):
    total = 0.0
    for i in range(len(pred)):
        total += len(set(pred[i][:k]) & set(gt[i][:k])) / k
    return total / len(pred)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latency", type=float, required=True)
    parser.add_argument("--jitter", type=float, required=True)
    parser.add_argument("--loss", type=float, required=True)
    parser.add_argument("--bandwidth", type=float, required=True)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--output", default="results/network_results.csv")

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    gt = np.load(GT_FILE)

    write_header = not os.path.exists(args.output)

    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "arch",
                "freshness",
                "latency_ms",
                "jitter_ms",
                "packet_loss",
                "bandwidth_mbps",
                "num_queries",
                "failures",
                "recall_at_10",
                "latency_p50_ms",
                "latency_p95_ms",
                "latency_p99_ms",
            ])

        for freshness, file in QUERY_FILES.items():

            queries = np.load(file)

            results = []
            latencies = []
            failures = 0
            valid_gt = []

            for i, q in enumerate(queries):

                t0 = time.time()
                response, failed = send_query(q)
                t1 = time.time()

                if failed:
                    failures += 1
                    continue

                _, I = response

                latencies.append((t1 - t0) * 1000.0)
                results.append(I[0])
                valid_gt.append(gt[i])  # keep alignment

            if len(results) == 0:
                r10 = 0.0
                p50 = p95 = p99 = 0.0
            else:
                results = np.array(results)
                valid_gt = np.array(valid_gt)

                r10 = recall_at_k(results, valid_gt, 10)

                p50 = np.percentile(latencies, 50)
                p95 = np.percentile(latencies, 95)
                p99 = np.percentile(latencies, 99)

            row = [
                args.arch,
                freshness,
                args.latency,
                args.jitter,
                args.loss,
                args.bandwidth,
                len(queries),
                failures,
                round(r10, 6),
                round(p50, 4),
                round(p95, 4),
                round(p99, 4),
            ]

            writer.writerow(row)
            print(row)


if __name__ == "__main__":
    main() 