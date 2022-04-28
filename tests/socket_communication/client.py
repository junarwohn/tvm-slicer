import socket
import numpy as np
import cv2
import struct
from multiprocessing import Process, Queue
import time

#HOST_IP = "127.0.0.1"
#HOST_IP = "172.31.59.132"
HOST_IP = "ec2-3-35-200-204.ap-northeast-2.compute.amazonaws.com"
#HOST_IP = socket.gethostbyaddr(HOST_IP)[0]
#HOST_IP = "3.35.200.204"
HOST_IP = "3.35.200.204"
#HOST_IP = "192.168.0.184"
PORT = 5000 
#PORT = 9998
PORT = 5050 

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print(HOST_IP, PORT)
client_socket.connect((HOST_IP, PORT))
print("connected")


def generate_img(q):
    cap = cv2.VideoCapture("../../tvm-slicer/src/data/j_scan.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        try:
            frame.resize((512,512))
        except:
            total_msg = struct.pack('i', 0)
            client_socket.sendall(total_msg)
            client_socket.close()
            break
        
        msg_body = b''
        # Send msg
        frame = np.array(frame)
        msg_body += frame.tobytes()
        q.put(frame)
        total_send_msg_size = len(msg_body)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body
        # Send object
        client_socket.sendall(send_msg)
    client_socket.close()

    
def recv_img(q):
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

        img_in_rgb = q.get()
        recv_msg = recv_msg[4*3*512*512:]

if __name__ == '__main__':
    q = Queue()
    p1 = Process(target=generate_img, args=(q,))
    p2 = Process(target=recv_img, args=(q,))
    stime = time.time()
    p1.start(); 
    p2.start(); 
    p1.join(); p2.join()
    print(time.time() - stime)

