import cv2

class OutputModule:
    def __init__(self, class_names=None):
        self.class_names = class_names if class_names else {0:"person",1:"car",2:"truck"}

    def draw(self, frame, tracks):
        """Draw bounding boxes + IDs on frame"""
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.tlbr)
            track_id = t.track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return frame

    def to_json(self, tracks):
        """Convert tracks to JSON-friendly format"""
        objs = []
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.tlbr)
            objs.append({
                "id": t.track_id,
                "bbox": [x1, y1, x2, y2],
                "class": self.class_names.get(int(t.cls), "unknown")
            })
        return objs
