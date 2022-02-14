from email import message_from_binary_file
import socket
import pickle
# import cloudpickle as pickle
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor
import json
import time
import sys
import cv2
import struct
from argparse import ArgumentParser
import ntplib 
from multiprocessing import Process

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.request(ntp_time_server, version=3) 

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')

args = parser.parse_args()

def get_time(is_enabled):
    if is_enabled == 1:
        return g_ntp_client.request(ntp_time_server, version=3).tx_time
    elif is_enabled == 0:
        return time.time()
    else:
        return 0

def make_preprocess(model, im_sz):
    if model == 'unet':
        def preprocess(img):
            return cv2.resize(img[490:1800, 900:2850], (im_sz,im_sz)).astype(np.float32) / 255
        return preprocess
    elif model == 'resnet152':
        def preprocess(img):
            return cv2.resize(img, (im_sz, im_sz))
        return preprocess

preprocess = make_preprocess(args.model, args.img_size)

# Model load

if args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
elif args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'opencl':
    target = 'opencl'
    dev = tvm.opencl()

# Video Load
img_size = 512 

HOST_IP = args.ip
PORT = 9998       
#socket_size = 16 * 1024 * 1024
socket_size = args.socket_size

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST_IP, PORT))

# initialize final output size
total_recv_bytes = struct.unpack('i', client_socket.recv(4))[0]
recv_msg = client_socket.recv(total_recv_bytes)
while len(recv_msg) < total_recv_bytes:
    recv_msg += client_socket.recv(total_recv_bytes)

final_output_shape = np.frombuffer(recv_msg, np.int).reshape((4,))

print(final_output_shape)

org=(50,100)
font=cv2.FONT_HERSHEY_SIMPLEX


def generate_img():
    cap = cv2.VideoCapture("../src/data/j_scan.mp4")
    timer_exclude_network = 0
    while (cap.isOpened()):
        timer_exclude_network_start = time.time()
        ret, frame = cap.read()
        try:
            frame = preprocess(frame)
        except:
            total_msg = struct.pack('i', 0)
            client_socket.sendall(total_msg)
            client_socket.close()
            break
        timer_exclude_network += time.time() - timer_exclude_network_start
        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
        msg_body = input_data.tobytes()

        total_send_msg_size = len(msg_body)
        send_msg = struct.pack('i', total_send_msg_size) + msg_body
        # Send object
        client_socket.sendall(send_msg)
        # print("send")
    client_socket.close()
    print("Exclude network time :", timer_exclude_network)

    
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
            # print(len(recv_msg))
            recv_msg += client_socket.recv(total_recv_msg_size)

        b,c,h,w = final_output_shape
        ## TODO : get output and parse 
        out = np.frombuffer(recv_msg[:4*b*c*h*w], np.float32).reshape(tuple(final_output_shape))
        th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.5, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))
        # img_in_rgb = frame
        # cv2.imshow("received - client", 255 * th)
        # cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
        recv_msg = recv_msg[4*b*c*h*w:]

    cv2.destroyAllWindows()

if __name__ == '__main__':
    timer_total_time = time.time()
    p1 = Process(target=generate_img)
    p2 = Process(target=recv_img)
    p1.start(); 
    p2.start(); 
    p1.join(); p2.join()
    print("Total time : ", time.time() - timer_total_time)

