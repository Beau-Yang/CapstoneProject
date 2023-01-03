import sys
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

code_dir_path = ROOT.joinpath("Code")
data_dir_path = ROOT.joinpath("Data")

cfg_path = code_dir_path.joinpath("yolov7.cfg")
weight_path = code_dir_path.joinpath("yolov7.weights")
image_path = data_dir_path.joinpath("LP.png") # LP.png frame.jpg

net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weight_path))
frame = cv2.imread(str(image_path))

# ?: size
# preprocess
blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(320, 320), mean=[0, 0, 0], swapRB=True)

# set input; set layer; forward; postprocess
net.setInput(blob)
layers_names = net.getLayerNames()
layers = [(layers_names[i - 1]) for i in net.getUnconnectedOutLayers()]
outputs = net.forward(layers)

vehicle_classes = [2, 3, 5, 7]
vehicle_boxes = []
vehicle_scores = []
vehicles = []

vehicle_conf_thr = 0.5
vehicle_nms_thr = 0.2


height, width = frame.shape[:2]

for output in outputs:
    for detection in output:
        # [x, y, w, h, conf, score1, score2, ..., score80]
        scores = detection[5:]
        class_id = np.argmax(scores)
        conf = scores[class_id]
        if (class_id in vehicle_classes) and (conf > vehicle_conf_thr):
            x = detection[0] * width
            y = detection[1] * height
            w = detection[2] * width
            h = detection[3] * height
            xmin, xmax = x - w / 2, x + w / 2
            ymin, ymax = y - h / 2, y + h / 2

            vehicle_boxes.append([xmin, xmax, ymin, ymax])
            vehicle_scores.append(float(conf))

vehicle_indices = cv2.dnn.NMSBoxes(vehicle_boxes, vehicle_scores, vehicle_conf_thr, vehicle_nms_thr)

for index in vehicle_indices:

    xmin, xmax, ymin, ymax = map(int, vehicle_boxes[index])
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 5)
    cv2.imshow("test", frame)
    cv2.waitKey()


"""
# face detection
prototxt_path = "Code/prototxt.txt"
model_path = "Code/res10_300x300_ssd_iter_140000_fp16.caffemodel" 
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
"""

# blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
