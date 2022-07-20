from multiprocessing import Process, Queue
import time

def background(data_queue):
    while True:
        if not data_queue.empty():
            data = data_queue.get()
            if len(data) == 0:
                break
            else:
                time.sleep(3)
                print("input :", data) 
    
if __name__ == '__main__':
    data_queue = Queue()
    p0 = Process(target=background, args=(data_queue,))
    p0.start()
    while True:
        cmd = input()
        data_queue.put(cmd)
        time.sleep(2)
        if cmd == '':
            break
    p0.join()
    print("end")