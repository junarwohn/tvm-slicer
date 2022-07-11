import socket
import pickle
import numpy as np
import cv2
import struct
from multiprocessing import Process, Queue
import time
import asyncio
from argparse import ArgumentParser

HOST_IP = "192.168.0.184"
PORT = 9998
parser = ArgumentParser()
parser.add_argument('--size', '-s', type=int, default=786432)
args = parser.parse_args()

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print(HOST_IP, PORT)
client_socket.connect((HOST_IP, PORT))
print("connected")


def send_img(data_q):
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
        print(total_send_msg_size)
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
        while len(recv_msg) < total_recv_msg_size:
            recv_msg += client_socket.recv(total_recv_msg_size)
        msg_data = recv_msg[:total_recv_msg_size]
        recv_msg = recv_msg[total_recv_msg_size:]
        data = pickle.loads(msg_data)

if __name__ == '__main__':
    data_q = [np.random.normal(0, 1, (args.size).astype(np.float32)) for i in range(253)]
    p1 = Process(target=send_img, args=(data_q,))
    p2 = Process(target=recv_img)
    p1.start(); 
    p2.start(); 
    stime = time.time()
    p1.join(); p2.join()
    print(time.time() - stime)

