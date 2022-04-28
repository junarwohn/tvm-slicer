import socket
import struct
import numpy as np

HOST_IP = "127.0.0.1" 
HOST_IP = "192.168.0.184" 
PORT = 9998        

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST_IP, PORT))

server_socket.listen()
client_socket, addr = server_socket.accept()

recv_msg = b''

while True:
    try:
        while len(recv_msg) < 4:
            recv_msg += client_socket.recv(4)
        total_recv_msg_size = struct.unpack('i', recv_msg[:4])[0]
        recv_msg = recv_msg[4:]
        if total_recv_msg_size == 0:
            client_socket.sendall(struct.pack('i', 0))
            break
    except:
        break

    while len(recv_msg) < total_recv_msg_size:
        # packet = client_socket.recvall()
        recv_msg += client_socket.recv(total_recv_msg_size)
 
    arr = np.random.normal(0, 1, (3, 512, 512)).astype(np.float32)
    send_msg = arr.tobytes()
    client_socket.sendall(struct.pack('i', len(send_msg)) + send_msg)
    recv_msg = recv_msg[total_recv_msg_size:]

client_socket.close()
server_socket.close()
