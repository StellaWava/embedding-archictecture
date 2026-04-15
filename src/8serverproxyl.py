"""
Step 8: Network server proxy for local 
testing. 
Wrapping FAISS as a simple PRC server. 

"""

import socket
import pickle
import faiss
import numpy as np

# HOST = "127.0.0.1"
# PORT = 5000

# INDEX_FILE = "data/hnsw.index"   # swap to ivf if needed
# INDEX_FILE1 = "data/ivf_flat.index"
# TOPK = 100
# BUFFER_SIZE = 4096

# print("Loading index...")
# index = faiss.read_index(INDEX_FILE)
# index1 = faiss.read_index(INDEX_FILE1)

# server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# server_sock.bind((HOST, PORT))
# server_sock.listen(5)

# print(f"ANN server listening on {HOST}:{PORT}")

# while True:
#     conn, addr = server_sock.accept()
#     try:
#         data = b""
#         while True:
#             chunk = conn.recv(BUFFER_SIZE)
#             if not chunk:
#                 break
#             data += chunk

#         if not data:
#             conn.close()
#             continue

#         payload = pickle.loads(data)
#         query = np.asarray(payload["query"], dtype="float32").reshape(1, -1)

#         # optional (if you want to use them later)
#         latency   = payload.get("latency")
#         jitter    = payload.get("jitter")
#         bandwidth = payload.get("bandwidth")

#         arch      = payload.get("arch")
#         knob      = payload.get("knob")

#         if arch == "hnsw":
#             index.hnsw.efSearch = knob
#         elif arch == "ivf":
#             index1.nprobe = knob

#         D, I = index.search(query, TOPK)

#         response = pickle.dumps((D, I))
#         conn.sendall(response)

#     except Exception as e:
#         print(f"Error handling client {addr}: {e}")
#     finally:
#         conn.close() 


HOST = "127.0.0.1"
PORT = 5000

HNSW_INDEX_FILE = "data/hnsw.index"
IVF_INDEX_FILE = "data/ivf_flat.index"

TOPK = 100
BUFFER_SIZE = 4096

print("Loading indexes...")
hnsw_index = faiss.read_index(HNSW_INDEX_FILE)
ivf_index = faiss.read_index(IVF_INDEX_FILE)

server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_sock.bind((HOST, PORT))
server_sock.listen(5)

print(f"ANN server listening on {HOST}:{PORT}")

while True:
    conn, addr = server_sock.accept()
    try:
        data = b""
        while True:
            chunk = conn.recv(BUFFER_SIZE)
            if not chunk:
                break
            data += chunk

        if not data:
            conn.close()
            continue

        payload = pickle.loads(data)
        query = np.asarray(payload["query"], dtype="float32").reshape(1, -1)

        arch = payload.get("arch")
        knob = payload.get("knob")

        if arch == "hnsw":
            hnsw_index.hnsw.efSearch = int(knob)
            D, I = hnsw_index.search(query, TOPK)

        elif arch == "ivf":
            ivf_index.nprobe = int(knob)
            D, I = ivf_index.search(query, TOPK)

        else:
            raise ValueError(f"Unsupported arch: {arch}")

        response = pickle.dumps((D, I))
        conn.sendall(response)

    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        conn.close()