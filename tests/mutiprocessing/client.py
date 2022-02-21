from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Lock
import socket
import time

from multiprocessing import Process, Value, Array, Lock, Queue
import time
import os
import numpy as np
import random

def enq(queue,l):
    for i in range(1000):
        # with l:
        queue.put([0, np.ones((3,1,512,512))])
        print('enq size', queue.qsize(), i)
        time.sleep(random.randint(0,99) / 1000)

    print("enq end")
    queue.put([])

    return 0

def deq(queue,l):
    while True:
        # print("[deq] acq")
        # if queue.qsize() != 0:
        # with l:
        if queue.qsize() != 0:
            data = queue.get()
            if len(data) == 0:
                break
            shape = data[1].shape
            size = queue.qsize()
        # print("[deq] rel")
            print('deq', shape, size)
            
        time.sleep(random.randint(0,99) / 10000)
    print("deq end")
    return 0

if __name__ == "__main__":

    numbers = range(1, 6)
    q = Queue()
    lock = Lock()
    p1 = Process(target=enq, args=(q,lock))
    p2 = Process(target=deq, args=(q,lock))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    while not q.empty():
        print("not empty")
        print(q.get())
    q.close()