import socket
import pickle
import faiss
import numpy as np

HOST = "127.0.0.1"
PORT = 5000

INDEX_FILE = "data/hnsw.index"   # swap to ivf if needed
TOPK = 100

print("Loading index...")
index = faiss.read_index(INDEX_FILE)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(5)

print(f"ANN server listening on {HOST}:{PORT}")

while True:
    conn, addr = sock.accept()
    data = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk

    query = pickle.loads(data)  # numpy array [d]

    query = query.astype("float32").reshape(1, -1)
    D, I = index.search(query, TOPK)

    response = pickle.dumps((D, I))
    conn.sendall(response)
    conn.close()