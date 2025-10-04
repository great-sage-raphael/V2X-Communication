# src/perception/enhanced_tracking_module.py
"""
Enhanced tracking module that integrates SUMO vehicle data
with traditional tracking algorithms for improved V2X performance.
"""

from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
import math
from types import SimpleNamespace


class EnhancedTracker:
    def __init__(self, frame_rate=30, track_thresh=0.6, track_buffer=30, 
                 match_thresh=0.8, use_sumo_matching=True):

        args = SimpleNamespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh
        )

        # Pass args + frame_rate (correct API usage)
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

        self.use_sumo_matching = use_sumo_matching
        self.vehicle_history = {}
        self.track_to_sumo_mapping = {}
        self.frame_count = 0