from cv2 import minAreaRect
import numpy as np

def quant(num):
    maxval = np.max(num)
    minval = np.min(num)
    delta = maxval - minval
    mean = (maxval + minval) / 2
    num = ((num - mean) / delta) * 128
    num = np.round(num)
    num = np.clip(num, 128, -127)
    return num

def dequant(num):
    num += 128
    