import socket
import threading
import time
import random

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 6000

TARGET_HOST = "127.0.0.1"
TARGET_PORT = 5000

LATENCY_MS = 0
JITTER_MS = 0
PACKET_LOSS = 0
BANDWIDTH_MBPS = 0

BUFFER_SIZE = 4096
BYTES_PER_SEC = BANDWIDTH_MBPS * 1024 * 1024 / 8 if BANDWIDTH_MBPS > 0 else None


def throttle_send(sock, data):
    for i in range(0, len(data), BUFFER_SIZE):
        chunk = data[i:i + BUFFER_SIZE]
        sock.sendall(chunk)
        if BYTES_PER_SEC:
            time.sleep(len(chunk) / BYTES_PER_SEC)


def recv_until_eof(sock):
    data = b""
    while True:
        chunk = sock.recv(BUFFER_SIZE)
        if not chunk:
            break
        data += chunk
    return data


def handle_client(client_socket):
    server_socket = None
    try:
        if random.random() < PACKET_LOSS:
            client_socket.close()
            return

        delay = LATENCY_MS / 1000.0 + random.uniform(-JITTER_MS, JITTER_MS) / 1000.0
        time.sleep(max(delay, 0))

        request = recv_until_eof(client_socket)
        if not request:
            client_socket.close()
            return

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.connect((TARGET_HOST, TARGET_PORT))

        throttle_send(server_socket, request)
        server_socket.shutdown(socket.SHUT_WR)

        response = recv_until_eof(server_socket)

        throttle_send(client_socket, response)

    except Exception as e:
        print(f"Proxy error: {e}")
    finally:
        try:
            client_socket.close()
        except Exception:
            pass
        if server_socket is not None:
            try:
                server_socket.close()
            except Exception:
                pass


proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
proxy_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
proxy_sock.bind((PROXY_HOST, PROXY_PORT))
proxy_sock.listen(5)

print(f"Proxy running on {PROXY_HOST}:{PROXY_PORT}")

while True:
    client, addr = proxy_sock.accept()
    threading.Thread(target=handle_client, args=(client,), daemon=True).start()