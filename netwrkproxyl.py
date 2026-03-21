import socket
import threading
import time
import random

# CONFIG
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 6000

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 5000

LATENCY_MS = 10
JITTER_MS = 5
PACKET_LOSS = 0.01
BANDWIDTH_MBPS = 100  # approximate

BYTES_PER_SEC = BANDWIDTH_MBPS * 1024 * 1024 / 8


def throttle_send(sock, data):
    chunk_size = 4096
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        sock.sendall(chunk)
        time.sleep(len(chunk) / BYTES_PER_SEC)


def handle_client(client_socket):
    if random.random() < PACKET_LOSS:
        client_socket.close()
        return

    delay = LATENCY_MS / 1000.0 + random.uniform(-JITTER_MS, JITTER_MS) / 1000.0
    time.sleep(max(delay, 0))

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.connect((TARGET_HOST, TARGET_PORT))

    data = b""
    while True:
        chunk = client_socket.recv(4096)
        if not chunk:
            break
        data += chunk

    throttle_send(server_socket, data)

    response = b""
    while True:
        chunk = server_socket.recv(4096)
        if not chunk:
            break
        response += chunk

    throttle_send(client_socket, response)

    client_socket.close()
    server_socket.close()


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((PROXY_HOST, PROXY_PORT))
sock.listen(5)

print(f"Proxy running on {PROXY_HOST}:{PROXY_PORT}")

while True:
    client, addr = sock.accept()
    threading.Thread(target=handle_client, args=(client,), daemon=True).start()

