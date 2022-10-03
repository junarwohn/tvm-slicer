import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# from UNetKerasAS import UNet as UnetAS
from UNetKerasReshapePreserve_noauto import UNet as UnetAS
import tensorflow as tf
# from UNetKerasOriginal import UNet as UnetAS


print("****************************")
model_as = UnetAS(3, 1, 64,[0,0,0,0])
# model_as = UnetAS(3, 1, 64)
# model_as.build(input_shape=(1,512,512,3))
model_as.build(input_shape=(1,256, 256, 3))
print(model_as.summary())
tf.keras.backend.clear_session()

print("****************************")
model_as = UnetAS(3, 1, 64,[1,0,0,0])
# model_as = UnetAS(3, 1, 64)
# model_as.build(input_shape=(1,512,512,3))
model_as.build(input_shape=(1,256, 256, 3))
print(model_as.summary())
tf.keras.backend.clear_session()

print("****************************")

model_as = UnetAS(3, 1, 64,[2,0,0,0])
# model_as = UnetAS(3, 1, 64)
# model_as.build(input_shape=(1,512,512,3))
model_as.build(input_shape=(1,256, 256, 3))
print(model_as.summary())
print("****************************")
