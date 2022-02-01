import cv2
import os

src_path = "./resnet_data/"
pp = os.listdir(src_path)
print(pp)

for i, img_path in enumerate(pp):
    frame = cv2.resize(cv2.imread(src_path + img_path), (512,512))
    cv2.imshow("received - client", frame)
    cv2.waitKey(1) 
    cv2.imwrite("frames/{}.jpg".format(i), frame)
