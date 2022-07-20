from http import client
import re
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

ntp_time_server = 'time.windows.com'               # NTP Server Domain Or IP 
ntp_time_server = 'time.google.com'               # NTP Server Domain Or IP 
g_ntp_client = ntplib.NTPClient() 
#response = c.eequest(timeServer, version=3) 

parser = ArgumentParser()
parser.add_argument('--start_point', '-s', type=int, default=0)
parser.add_argument('--end_point', '-e', type=int, default=-1)
parser.add_argument('--partition_point', '-p', type=int, default=0, help='set partition point')
parser.add_argument('--img_size', '-i', type=int, default=512, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=2, help='set opt_level')
parser.add_argument('--ip', type=str, default='192.168.0.184', help='input ip of host')
parser.add_argument('--socket_size', type=int, default=1024*1024, help='socket data size')
parser.add_argument('--ntp_enable', type=int, default=0, help='ntp support')

args = parser.parse_args()

HOST_IP = args.ip
PORT = 9998        
#socket_size = 16 * 1024 * 1024 
socket_size = args.socket_size

client_socket  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
client_socket.connect((HOST_IP, PORT))
new_data = np.random.normal(0, 1, (3, args.img_size, args.img_size)).astype(np.float32)
time_start = time.time()
for i in range(1000):
    msg_body = pickle.dumps(new_data)
    total_send_msg_size = len(msg_body)
    send_msg = struct.pack('i', total_send_msg_size) + msg_body

    client_socket.sendall(send_msg)
    # print("send")
time_end = time.time() - time_start
client_socket.sendall(struct.pack('i', 0))
print("totaltime", time_end, "data size", len(send_msg))
client_socket.close()
