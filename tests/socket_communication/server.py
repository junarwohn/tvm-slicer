import pickle
import socket
import struct
import numpy as np
from multiprocessing import Process, Queue
import time

HOST_IP = "192.168.0.184" 
PORT = 9998        

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST_IP, PORT))

server_socket.listen()
client_socket, addr = server_socket.accept()
arr = np.random.normal(0, 1, (1, 512, 512)).astype(np.float32)

def recv_img(q):
    recv_msg = b''
    while True:
        while len(recv_msg) < 4:
            recv_msg += client_socket.recv(4)
        total_recv_msg_size = struct.unpack('i', recv_msg[:4])[0]
        recv_msg = recv_msg[4:]
        if total_recv_msg_size == 0:
            q.put([])
            break 
        while len(recv_msg) < total_recv_msg_size:
            recv_msg += client_socket.recv(total_recv_msg_size)
        msg_data = recv_msg[:total_recv_msg_size]
        recv_msg = recv_msg[total_recv_msg_size:]
        data = pickle.loads(msg_data)
        q.put(arr)
        
def send_img(q):
    while True:
        try:
            frame = q.get()
            if len(frame) == 0:
                total_msg = struct.pack('i', 0)
                client_socket.sendall(total_msg)
                client_socket.close()
                break
        except:
            total_msg = struct.pack('i', 0)
            client_socket.sendall(total_msg)
            client_socket.close()
            break
        
        # Send msg
        msg_body = pickle.dumps(frame)
        total_send_msg_size = len(msg_body)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body
        # Send object
        client_socket.sendall(send_msg)
    client_socket.close()


if __name__ == '__main__':
    q = Queue()
    p1 = Process(target=send_img, args=(q,))
    p2 = Process(target=recv_img, args=(q,))
    stime = time.time()
    p1.start(); 
    p2.start(); 
    p1.join(); p2.join()
    print(time.time() - stime)

    client_socket.close()
    server_socket.close()
