from ultralystics import YOLO
import numpy as np

class DetectionModule:
    def __init__(self,model_path="yolov8n.pt"):
        self.model=YOLO(model_path)
    def detect(self,frame):
        results=self.model(frame)[0]
        detections=[]
        for box in results.boxes:
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            detections.append([x1,y1,x2,y2,conf,cls])
        return np.array(detections)