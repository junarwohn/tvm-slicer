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
from multiprocessing import Process, Queue, Lock

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


model_info_path = "../src/graph/{}_{}_front_{}_{}_{}.json".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
with open(model_info_path, "r") as json_file:
    model_info = json.load(json_file)

input_info = model_info["extra"]["inputs"]
shape_info = model_info["attrs"]["shape"][1][:len(input_info)]
output_info = model_info["extra"]["outputs"]

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

def read_and_infernece(visual_queue, visual_lock, send_queue, send_lock):
    model_path = "../src/model/{}_{}_front_{}_{}_{}.so".format(args.model, args.target, args.img_size, args.opt_level, args.partition_point)
    front_lib = tvm.runtime.load_module(model_path)
    front_model = graph_executor.GraphModule(front_lib['default'](dev))

    cap = cv2.VideoCapture("../src/data/j_scan.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        try:
            frame = preprocess(frame)
        # no more frame to read
        except:
            # visual_lock.acquire()
            visual_queue.put([])
            # visual_lock.release()
            # send_lock.acquire()
            send_queue.put([])
            # send_lock.release()
            print('read_and_infernece end')
            break

        # visual_lock.acquire()
        visual_queue.put(frame)
        # visual_lock.release()
        input_data = np.expand_dims(frame, 0).transpose([0, 3, 1, 2])
        front_model.set_input("input_0", input_data)
        front_model.run()
        # send_lock.acquire()
        
        outs = []
        for i, out_idx in enumerate(output_info):
            out = front_model.get_output(i).asnumpy().astype(np.float32)
            outs.append(out)
        
        # outs = []
        # for i, out_idx in enumerate(output_info):
        #     out = front_model.get_output(i).asnumpy().astype(np.float32)
        
        send_queue.put(outs)
        # send_lock.release()
        # print("read_and_infernece")
    cap.release()


def send_data(send_queue, send_lock):
    # Send msg
    output_num = len(output_info)
    while True:
        #if send_queue.qsize() != 0:
        if True:
            # outs = []
            # # send_lock.acquire()
            # for i in range(output_num):
            #     outs.append(send_queue.get())
            # send_lock.release()
            outs = send_queue.get()
            # inference end
            if len(outs) == 0:
                total_msg = struct.pack('i', 0)
                client_socket.sendall(total_msg)
                while send_queue.qsize() != 0:
                    send_queue.get()
                send_queue.close()
                print("send_data end")
                break

            msg_body = b''
            for out in outs:
                out_byte = out.tobytes()
                msg_body += out_byte

            total_send_msg_size = len(msg_body)
            send_msg = struct.pack('i', total_send_msg_size) + msg_body
            client_socket.sendall(send_msg)
            # print("send_data")

        else:
            # yield to other process
            time.sleep(0)
    # send_queue.close()


def recv_data(result_queue, result_lock):
    recv_msg = b''
    while True:
        while len(recv_msg) < 4:
            recv_msg += client_socket.recv(4)
        total_recv_msg_size = struct.unpack('i', recv_msg[:4])[0]
        recv_msg = recv_msg[4:]
        
        if total_recv_msg_size == 0:
            # result_lock.acquire()
            print("recv end")
            result_queue.put([])
            # result_lock.release()
            break 

        while len(recv_msg) < total_recv_msg_size:
            recv_msg += client_socket.recv(total_recv_msg_size)

        b,c,h,w = final_output_shape
        out = np.frombuffer(recv_msg[:4*b*c*h*w], np.float32).reshape(tuple(final_output_shape))
        th = cv2.resize(cv2.threshold(np.squeeze(out.transpose([0,2,3,1])), 0.8, 1, cv2.THRESH_BINARY)[-1], (img_size,img_size))

        # result_lock.acquire()
        result_queue.put(th)
        # result_lock.release()

        recv_msg = recv_msg[4*b*c*h*w:]

def visualize(visual_queue, visual_lock, result_queue, result_lock):
    cnt = 0
    while True:
        #if result_queue.qsize() != 0:
        if True:
            # result_lock.acquire()
            th = result_queue.get()
            # result_lock.release()
            # visual_lock.acquire()
            img_in_rgb = visual_queue.get()
            # visual_lock.release()
            if len(img_in_rgb) == 0 or len(th) == 0:
                while result_queue.qsize() != 0:
                    result_queue.get()
                while visual_queue.qsize() != 0:
                    visual_queue.get()
                print("visualize end")
                break
            #img_in_rgb[th == 1] = [0, 0, 255]
            #cv2.imshow("received - client", img_in_rgb)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            #print("visualize")
        else:
            time.sleep(0)
    # visual_queue.close()
    # result_queue.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    visual_queue = Queue()
    visual_lock = Lock()
    send_queue = Queue()
    send_lock = Lock() 
    result_queue = Queue()
    result_lock = Lock()

    p_read_and_infernece = Process(target=read_and_infernece, args=(visual_queue, visual_lock, send_queue, send_lock))
    p_send_data = Process(target=send_data, args=(send_queue, send_lock))
    p_recv_data = Process(target=recv_data, args=(result_queue, result_lock))
    p_visualize = Process(target=visualize, args=(visual_queue, visual_lock, result_queue, result_lock))

    p_read_and_infernece.start()
    p_send_data.start()
    p_recv_data.start()
    p_visualize.start()
    stime = time.time()
    p_read_and_infernece.join()
    print("join p_read_and_infernece")
    p_send_data.join()
    print("join p_send_data")
    p_recv_data.join()
    print("join p_recv_data")
    p_visualize.join()
    print("join p_visualize")
    print(time.time() - stime)
    client_socket.close()
