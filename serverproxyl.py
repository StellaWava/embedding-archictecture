import socket
import pickle
import faiss
import numpy as np

HOST = "127.0.0.1"
PORT = 5000

INDEX_FILE = "data/hnsw.index"   # swap to ivf if needed
TOPK = 100
BUFFER_SIZE = 4096

print("Loading index...")
index = faiss.read_index(INDEX_FILE)

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

        query = pickle.loads(data)
        query = np.asarray(query, dtype="float32").reshape(1, -1)

        D, I = index.search(query, TOPK)

        response = pickle.dumps((D, I))
        conn.sendall(response)

    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        conn.close() 