# src/perception/sumo_to_detections.py
"""
Convert SUMO vehicle list -> tracker detections.

We build a simple top-down mapping:
 - choose an origin and scale to map SUMO meters -> pixels
 - each vehicle becomes a small bbox centered at (x_px, y_px) with a fixed size (or proportional to vehicle length)

Detection format returned: list of [x1, y1, x2, y2, conf, class]
"""

import math
import numpy as np

class SumoToDetections:
    def __init__(self, world_bbox=None, image_size=(800, 800), default_bbox_size_m=2.0):
        """
        world_bbox: (xmin, ymin, xmax, ymax) in SUMO meters that define map area to visualize/map to image.
                    If None, we compute bounds dynamically from data (but stable mapping is better).
        image_size: (width_px, height_px) output pixel resolution for top-down mapping.
        default_bbox_size_m: default vehicle length in meters -> used to compute bbox size in px.
        """
        self.world_bbox = world_bbox
        self.image_w, self.image_h = image_size
        self.default_bbox_size_m = default_bbox_size_m
        self._computed = False

    def compute_world_bounds(self, vehicles):
        """Compute bounding box from current vehicle positions (if world_bbox not provided)."""
        xs = [v["x"] for v in vehicles]
        ys = [v["y"] for v in vehicles]
        margin = 10.0  # meters margin
        self.world_bbox = (min(xs)-margin, min(ys)-margin, max(xs)+margin, max(ys)+margin)
        self._computed = True

    def world_to_pixel(self, x, y):
        """Map (x,y) in meters to (px_x, px_y). origin at top-left pixel (0,0)."""
        xmin, ymin, xmax, ymax = self.world_bbox
        # Normalize in [0,1]
        nx = (x - xmin) / (xmax - xmin)
        ny = (y - ymin) / (ymax - ymin)
        # pixel coordinates: flip y so increasing SUMO y maps downwards in image
        px = int(nx * (self.image_w - 1))
        py = int((1 - ny) * (self.image_h - 1))
        return px, py

    def vehicles_to_detections(self, vehicles, conf=1.0, class_id=1):
        """
        vehicles: list of dicts from SUMO
        returns: np.array of shape (N,6): [x1,y1,x2,y2,conf,class_id]
        """
        if not self.world_bbox:
            self.compute_world_bounds(vehicles)

        detections = []
        for v in vehicles:
            x, y = v["x"], v["y"]
            px, py = self.world_to_pixel(x, y)
            # bbox size in pixels: proportional to default vehicle length
            # convert default_bbox_size_m (meters) -> pixels
            xmin, ymin, xmax, ymax = self.world_bbox
            meters_to_px = self.image_w / (xmax - xmin)
            half_w = int((self.default_bbox_size_m * meters_to_px) / 2)
            half_h = int(half_w * 0.6)  # approximate vehicle height smaller than length
            x1 = px - half_w
            y1 = py - half_h
            x2 = px + half_w
            y2 = py + half_h
            # optional clamp to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(self.image_w-1, x2); y2 = min(self.image_h-1, y2)
            detections.append([x1, y1, x2, y2, conf, class_id])
        return np.array(detections, dtype=float)
