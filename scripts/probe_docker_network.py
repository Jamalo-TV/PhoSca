import socket

for i in range(2, 10):
    ip = f"172.18.0.{i}"
    for port in (5432, 6379, 8000, 8080):
        sock = socket.socket()
        sock.settimeout(0.2)
        try:
            sock.connect((ip, port))
            print(ip, port, "open")
        except Exception:
            pass
        finally:
            sock.close()

