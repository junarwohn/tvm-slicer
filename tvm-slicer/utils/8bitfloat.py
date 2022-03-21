import numpy as np

# input for np.float32
def to_8bit(num):
    num = num.view(np.uint32)
    #print(0, num)
    print(0, np.vectorize(np.binary_repr)(num, width=32))
    front = int('0b11111000000000000000000000000000', 2)
    back  = int('0b00000000001110000000000000000000', 2)

    f = (num & front) >> 24
    #print(0, np.vectorize(np.binary_repr)(f, width=32))
    b = (num & back) >> 19
    #print(0, np.vectorize(np.binary_repr)(b, width=32))
    t = f | b
    print(0, np.vectorize(np.binary_repr)(t, width=32))

    return t.astype(np.uint8)

def to_32bit(num):
    #print(0, np.vectorize(np.binary_repr)(num, width=32))
    num = num.astype(np.uint32)
    #print(0, np.vectorize(np.binary_repr)(num, width=32))
    front = int('0b11111000', 2)
    back  = int('0b00000111', 2)

    f = num & front 
    b = num & back
    t = ((f << 5) | b) << 19
    print(0, np.vectorize(np.binary_repr)(t, width=32))
    
    return t.view(np.float32)

if __name__ == '__main__':
    #arr = (np.random.randint(0,100,(4)) / 100).astype(np.float32)
    arr = (np.array([0.5,0.25])).astype(np.float32)
    print(arr)

    #arr = np.array([int('0xffffffff', 16), int('0xffffffff', 16)]).astype(np.uint32)
    f8 = to_8bit(arr)
    f32 = to_32bit(f8)
    print(f32.astype(np.float32))
