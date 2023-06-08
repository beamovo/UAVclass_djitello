import cv2
import numpy as np
import os

#neural network 改成网络参数文件位置
net = cv2.dnn.readNetFromCaffe(r"E:\Coding\UAVclass\TT_demo\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master/MobileNetSSD_deploy.prototxt.txt",
							   r"E:\Coding\UAVclass\TT_demo\Human-Detection-and-Tracking-through-the-DJI-Tello-minidrone-master/MobileNetSSD_deploy.caffemodel") #modify with the NN path
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

img = cv2.imread('7.jpg')

blob = cv2.dnn.blobFromImage(img, 0.007843, (180, 180), (0, 0, 0), True, crop=False)

net.setInput(blob)
detections = net.forward()

h = img.shape[0]
w = img.shape[1]
for i in np.arange(0, detections.shape[2]):
	idx = int(detections[0, 0, i, 1])
	confidence = detections[0, 0, i, 2]

	if CLASSES[idx] != "background":
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		cv2.rectangle(img, (startX, startY), (endX, endY), colors[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
		cv2.imshow('detect', img)
cv2.waitKey(0)

