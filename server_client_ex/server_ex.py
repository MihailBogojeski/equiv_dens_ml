import socket

HOST = ''
PORT = 50007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()

for i in range(3):
    data = conn.recv(1024)
    conn.sendall(data)

s.close()
