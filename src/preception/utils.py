import cv2

def draw_boxes(frame, tracked_objects, class_names):
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        track_id = obj["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return frame
