import struct
import numpy as np
import time

target_data = list(range(51))
#target_data_2 = list(range(512*512*1))
print(len(target_data))

struct_time = time.time()
struct_data = []
for i in target_data:
    struct_data.append(struct.pack('i', i))

unstruct_data = []
for i in struct_data:
    unstruct_data.append(struct.unpack('i', i)[0])
print(time.time() - struct_time)

np_time = time.time()
np_data = []
for i in target_data:
    np_data.append(np.array(i).tobytes())

unnp_data = []
for i in np_data:
    np_data.append(np.frombuffer(i))
print(time.time() - np_time)


