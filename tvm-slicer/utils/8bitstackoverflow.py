# https://stackoverflow.com/questions/56962100/define-a-custom-float8-in-python-numpy-and-convert-from-to-float16

import numpy as np

def to_8bit(num):
    float16 = num.astype(np.float16) # Here's some data in an array
    float8s = float16.tobytes()[1::2]
    return float8s

def from_8bit(num):
    float16 = np.frombuffer(np.array(np.frombuffer(num, dtype='u1'), dtype='>u2').tobytes(), dtype='f2')
    return float16.astype(np.float32)

if __name__ == '__main__':
    arr = np.random.normal(0, 1, (1, 3, 4, 4)).astype(np.float32)
    print(arr)
    arr8 = to_8bit(arr)
    arr32 = from_8bit(arr8).reshape((1,3,4,4))
    print(arr32)

