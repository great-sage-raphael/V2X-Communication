from yolox.tracker.byte_tracker import BYTETracker
import numpy as np

class Tracker:
    def __init__(self, frame_rate=30):
        self.tracker = BYTETracker(frame_rate=frame_rate)

    def update(self, detections, frame):
        """
        detections: np.array [[x1,y1,x2,y2,conf,cls], ...]
        frame_shape: (height, width, channels)
        returns list of tracks with IDs
        """
        tracks = self.tracker.update(
            np.array(detections), [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]]
        )

        tracked = []
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            tracked.append({
                "track_id": t.track_id,
                "bbox": [x1, y1, x2, y2],
                "score": t.score
            })
        return tracked
