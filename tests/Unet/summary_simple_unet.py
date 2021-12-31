from simple_unet import UNet

model = UNet(in_dim=3, out_dim=1, num_filter=16)

model.summary()