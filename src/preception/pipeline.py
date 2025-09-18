import cv2
from detector import Detector
from tracker_modules import Tracker
from utils import draw_boxes

def run_pipeline(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = Detector("yolov8n.pt", conf_threshold=0.5)
    tracker = Tracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections, frame)

        frame = draw_boxes(frame, tracked_objects, class_names=[])
        cv2.imshow("YOLOv8 + ByteTrack", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
