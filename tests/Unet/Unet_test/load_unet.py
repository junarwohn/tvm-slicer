import tensorflow as tf
from tensorflow import keras 

model_config = [0, 0, 0, 0]
model = keras.models.load_model('./unet_as_{}_{}_{}_{}.h5'.format(*model_config))
for layer in model.layers:
    layer._name = 'input_1'
    break
print(model.summary())