import sys
from pathlib import Path

import cv2
import imutils
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

code_dir_path = ROOT.joinpath("Code")
data_dir_path = ROOT.joinpath("Data")

class BlurTool:
    def __init__(self):

        cfg_path = code_dir_path.joinpath("yolov7.cfg")
        weight_path = code_dir_path.joinpath("yolov7.weights")
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weight_path))
        self.layers = [(self.net.getLayerNames()[i - 1]) for i in self.net.getUnconnectedOutLayers()]
    
    def _check_file(self, file_name):

        self.file_path = data_dir_path.joinpath(file_name)
        assert self.file_path.exists(), "File does not exist!"

        suffix = self.file_path.suffix
        if suffix in [".png", ".jpg"]:
            self.is_video = False
        elif suffix in [".mp4"]:
            self.is_video = True
        else:
            raise ValueError(f"No Support for the Format: {suffix}")

    def process(self, file_name="LP.png"):

        self._check_file(file_name)
        if not self.is_video:
            self.process_image()
        else:
            self.process_video()

    def process_image(self):
        frame = cv2.imread(str(self.file_path))
        vehicles = self.detect_vehicle(frame)
        # for vehicle in vehicles:
        #     xmin, xmax, ymin, ymax = vehicle
        #     self.detect_license_plate(frame[ymin: ymax, xmin: xmax])

        xmin, xmax, ymin, ymax = vehicles[1]
        self.detect_license_plate(frame[ymin: ymax, xmin: xmax])

        # print(vehicles)

    def process_video(self):
        cap = cv2.VideoCapture()
    

    def detect_vehicle(self, frame):

        frame_height, frame_width = frame.shape[:2]

        vehicle_classes = [2, 3, 5, 7]
        vehicle_boxes, vehicle_scores, vehicles = [], [], []

        vehicle_conf_thr, vehicle_nms_thr = 0.5, 0.2
        # ?: size
        # preprocess -> set input -> forward
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(320, 320), mean=[0, 0, 0], swapRB=True)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layers)

        for output in outputs:
            for detection in output:
                # [x, y, w, h, conf, score1, score2, ..., score80]
                scores = detection[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if (class_id in vehicle_classes) and (conf > vehicle_conf_thr):
                    x, y = detection[0] * frame_width, detection[1] * frame_height
                    w, h = detection[2] * frame_width, detection[3] * frame_height
                    xmin, xmax = x - w / 2, x + w / 2
                    ymin, ymax = y - h / 2, y + h / 2

                    vehicle_boxes.append((xmin, xmax, ymin, ymax))
                    vehicle_scores.append(float(conf))

        #  postprocess: nms -> size filter
        vehicle_indices = cv2.dnn.NMSBoxes(vehicle_boxes, vehicle_scores, vehicle_conf_thr, vehicle_nms_thr)
        for index in vehicle_indices:
            xmin, xmax, ymin, ymax = map(int, vehicle_boxes[index])
            if (xmax - xmin) * (ymax - ymin) >= frame_width * frame_height * 0.03:
                # [(xmin, xmax, ymin, ymax), ...]
                vehicles.append((xmin, xmax, ymin, ymax))
        
        return vehicles

    def detect_license_plate(self, frame):
        
        license_plate = None
        # convert to grey scale -> reduce noise -> detect edges
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_reduce_noise = cv2.bilateralFilter(frame_gray, d=13, sigmaColor=15, sigmaSpace=15)
        frame_edge = cv2.Canny(frame_reduce_noise, threshold1=30, threshold2=200)

        contours = imutils.grab_contours(cv2.findContours(frame_edge.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE))
        # check which one has a rectangle shape (4 sides) and closed figure
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(curve=cnt, closed=True)
            approx = cv2.approxPolyDP(curve=cnt, epsilon=0.1 * peri, closed=True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                license_plate = (x, y, x + w, y + h)
                break
        
        return license_plate
        # # open -> threshold -> edge detection
        # frame_open = cv2.morphologyEx(frame_reduce_noise, op=cv2.MORPH_OPEN, kernel=np.ones((23, 23), np.uint8))
        # frame_add_weight = cv2.addWeighted(src1=frame_reduce_noise, alpha=1, src2=frame_open, beta=-1, gamma=0)
        # _, frame_thresh = cv2.threshold(frame_add_weight, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # frame_edge = cv2.Canny(frame_thresh, threshold1=100, threshold2=200)

        # frame_edge = cv2.morphologyEx(frame_edge, op=cv2.MORPH_CLOSE, kernel=np.ones((10, 10), np.uint8))
        # frame_edge = cv2.morphologyEx(frame_edge, cv2.MORPH_OPEN, kernel=np.ones((10, 10), np.uint8))

        # contours = imutils.grab_contours(cv2.findContours(frame_edge.copy() , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
        # for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        #     (x, y), (w, h), angle = cv2.minAreaRect(cnt)



        # cv2.imshow("test", frame_edge)
        # cv2.waitKey()
        # exit()


        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # cv2.imshow("test", frame)
        # cv2.waitKey()
        # cv2.drawContours(frame, [a], -1, (0, 255, 0), 3)
        # cv2.imshow("test", frame)
        # cv2.waitKey()

blur_tool = BlurTool()
blur_tool.process(file_name="LP.png") # LP.png frame.jpg


"""
# face detection
prototxt_path = "Code/prototxt.txt"
model_path = "Code/res10_300x300_ssd_iter_140000_fp16.caffemodel" 
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


xmin, xmax, ymin, ymax = map(int, vehicle_boxes[index])
vehicles.append()
cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 5)
cv2.imshow("test", frame)
cv2.waitKey()
"""
