import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing

bs = 1
num_class = 1000

img_shape = (3, 224, 224)

data_shape = (bs, ) + img_shape
output_shape = (bs, num_class)

