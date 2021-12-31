from simple_unet import UNet
import tensorflow as tf

model = UNet(in_dim=3, out_dim=1, num_filter=16)

model.build(input_shape=(16, 512, 512,3))
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['binary_crossentropy'])

model.save('unet_512.h5')