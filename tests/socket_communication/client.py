import socket
import pickle
import numpy as np
import cv2
import struct
from multiprocessing import Process, Queue
import time

HOST_IP = "192.168.0.184"
PORT = 9998

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print(HOST_IP, PORT)
client_socket.connect((HOST_IP, PORT))
print("connected")


def generate_img(q, data_q):
    while True:
        try:
            frame = data_q.pop(0)
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

    
def recv_img():
    recv_msg = b''
    while True:
        while len(recv_msg) < 4:
            # print("recv")
            recv_msg += client_socket.recv(4)
        total_recv_msg_size = struct.unpack('i', recv_msg[:4])[0]
        recv_msg = recv_msg[4:]
        if total_recv_msg_size == 0:
            break 
        # print("total_recv_msg_size", total_recv_msg_size)
        # recv_msg += client_socket.recv(total_recv_msg_size)
        while len(recv_msg) < total_recv_msg_size:
            # print(len(recv_msg))
            recv_msg += client_socket.recv(total_recv_msg_size)
        # img = np.frombuffer(recv_msg[:4*512*512*3], np.float32).reshape((512,512,3))
        msg_data = recv_msg[:total_recv_msg_size]
        recv_msg = recv_msg[total_recv_msg_size:]
        data = pickle.loads(msg_data)

if __name__ == '__main__':
    q = Queue()
    data_q = [np.random.normal(0, 1, (3, 512, 512)) for i in range(253)]
    p1 = Process(target=generate_img, args=(q, data_q))
    p2 = Process(target=recv_img)
    stime = time.time()
    p1.start(); 
    p2.start(); 
    p1.join(); p2.join()
    print(time.time() - stime)

