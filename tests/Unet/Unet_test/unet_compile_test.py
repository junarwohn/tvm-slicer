import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from UNetKerasAS import UNet as UnetAS

model_as = UnetAS(3, 1, 64,[3,0,0,0])
# model_as.build(input_shape=(1,512,512,3))
model_as.build(input_shape=(1,256, 256, 3))
print(model_as.summary())