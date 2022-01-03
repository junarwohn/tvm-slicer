import cv2
cap = cv2.VideoCapture("./j_scan.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
height, width = (512,512)
print("fps of vid:", fps)
g_cnt = 0
while (cap.isOpened()):
    #ret, frame = cap.read()
    ret, frame = cap.read()
    try:
        frame = cv2.resize(frame[490:1800, 900:2850], (height,width))
    except:
        # print(frame, ret)
        break
    if ret:
        cv2.imshow("asda", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
