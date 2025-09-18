# src/perception/enhanced_tracking_module.py
"""
Enhanced tracking module that integrates SUMO vehicle data
with traditional tracking algorithms for improved V2X performance.
"""

from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
import math

class EnhancedTracker:
    def __init__(self, frame_rate=30, track_thresh=0.6, track_buffer=30, 
                 match_thresh=0.8, use_sumo_matching=True):
        """
        Initialize enhanced tracker with SUMO integration.
        
        Args:
            frame_rate: Video frame rate
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Threshold for matching tracks
            use_sumo_matching: Whether to use SUMO data for track matching
        """
        self.tracker = BYTETracker(
            frame_rate=frame_rate,
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh
        )
        
        self.use_sumo_matching = use_sumo_matching
        self.vehicle_history = {}  # Store vehicle position history
        self.track_to_sumo_mapping = {}  # Map track IDs to SUMO vehicle IDs
        self.frame_count = 0
        
    def update(self, detections, frame, sumo_vehicles=None):
        """
        Update tracker with detections and optional SUMO vehicle data.
        
        Args:
            detections: np.array [[x1,y1,x2,y2,conf,cls], ...]
            frame: Current frame (for ByteTracker)
            sumo_vehicles: List of SUMO vehicle data dicts
            
        Returns:
            List of enhanced track objects with SUMO data integration
        """
        self.frame_count += 1
        
        # Run standard ByteTracker
        if len(detections) == 0:
            detections = np.empty((0, 6))
            
        byte_tracks = self.tracker.update(
            detections, 
            [frame.shape[0], frame.shape[1]], 
            [frame.shape[0], frame.shape[1]]
        )
        
        # Enhance tracks with SUMO data if available
        enhanced_tracks = []
        for track in byte_tracks:
            enhanced_track = self.enhance_track_with_sumo(track, sumo_vehicles)
            enhanced_tracks.append(enhanced_track)
        
        # Update vehicle history
        self.update_vehicle_history(enhanced_tracks)
        
        return enhanced_tracks
    
    def enhance_track_with_sumo(self, track, sumo_vehicles):
        """Enhance ByteTracker track with SUMO vehicle data."""
        if not sumo_vehicles or not self.use_sumo_matching:
            return self.create_enhanced_track_dict(track)
        
        # Try to match track with SUMO vehicle
        matched_vehicle = self.match_track_to_sumo_vehicle(track, sumo_vehicles)
        
        enhanced_track = self.create_enhanced_track_dict(track)
        
        if matched_vehicle:
            # Add SUMO data to track
            enhanced_track['sumo_id'] = matched_vehicle['id']
            enhanced_track['world_position'] = (matched_vehicle['x'], matched_vehicle['y'])
            enhanced_track['speed'] = matched_vehicle['speed']
            enhanced_track['angle'] = matched_vehicle['angle']
            enhanced_track['has_sumo_data'] = True
            
            # Update mapping
            self.track_to_sumo_mapping[track.track_id] = matched_vehicle['id']
            
            # Calculate additional metrics
            enhanced_track['acceleration'] = self.calculate_acceleration(track.track_id, matched_vehicle['speed'])
            enhanced_track['direction_vector'] = self.calculate_direction_vector(matched_vehicle['angle'])
            
        else:
            enhanced_track['has_sumo_data'] = False
        
        return enhanced_track
    
    def create_enhanced_track_dict(self, track):
        """Create enhanced track dictionary from ByteTracker track."""
        x1, y1, x2, y2 = track.tlbr
        
        return {
            'track_id': track.track_id,
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'score': float(track.score),
            'class': int(getattr(track, 'cls', 1)),
            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
            'frame_count': self.frame_count,
            'is_confirmed': track.is_activated,
            'track_len': track.tracklet_len,
            # Will be filled by SUMO data if available
            'sumo_id': None,
            'world_position': None,
            'speed': 0.0,
            'angle': 0.0,
            'acceleration': 0.0,
            'direction_vector': (0.0, 0.0),
            'has_sumo_data': False
        }
    
    def match_track_to_sumo_vehicle(self, track, sumo_vehicles):
        """
        Match ByteTracker track to SUMO vehicle using position and motion.
        """
        track_center = ((track.tlbr[0] + track.tlbr[2]) / 2, 
                       (track.tlbr[1] + track.tlbr[3]) / 2)
        
        # First check if we have an existing mapping
        if track.track_id in self.track_to_sumo_mapping:
            sumo_id = self.track_to_sumo_mapping[track.track_id]
            for vehicle in sumo_vehicles:
                if vehicle['id'] == sumo_id:
                    return vehicle
        
        # If no existing mapping, find best match by proximity
        best_match = None
        best_distance = float('inf')
        
        for vehicle in sumo_vehicles:
            # Convert world coordinates to pixel coordinates would be needed here
            # For now, use simplified matching based on relative positions
            vehicle_key = f"{vehicle['x']:.1f},{vehicle['y']:.1f}"
            
            # Simple distance-based matching (would need coordinate conversion in real scenario)
            # This is a placeholder - you'd need to use your SumoToDetections converter here
            distance = self.calculate_track_vehicle_distance(track_center, vehicle)
            
            if distance < best_distance and distance < 50:  # 50 pixel threshold
                best_distance = distance
                best_match = vehicle
        
        return best_match
    
    def calculate_track_vehicle_distance(self, track_center, vehicle):
        """Calculate distance between track center and vehicle position."""
        # This is simplified - in practice you'd convert SUMO coordinates to pixel coordinates
        # For now, use a mock distance calculation
        return np.random.uniform(0, 100)  # Placeholder
    
    def calculate_acceleration(self, track_id, current_speed):
        """Calculate acceleration from speed history."""
        if track_id not in self.vehicle_history:
            return 0.0
        
        history = self.vehicle_history[track_id]
        if len(history) < 2:
            return 0.0
        
        # Calculate acceleration from last two speed measurements
        prev_speed = history[-1].get('speed', 0)
        dt = 0.1  # Assuming 100ms time steps
        acceleration = (current_speed - prev_speed) / dt
        
        return acceleration
    
    def calculate_direction_vector(self, angle):
        """Calculate direction vector from angle."""
        angle_rad = math.radians(angle)
        return (math.cos(angle_rad), math.sin(angle_rad))
    
    def update_vehicle_history(self, enhanced_tracks):
        """Update vehicle movement history."""
        for track in enhanced_tracks:
            track_id = track['track_id']
            
            if track_id not in self.vehicle_history:
                self.vehicle_history[track_id] = []
            
            # Store current state
            state = {
                'frame': self.frame_count,
                'position': track['center'],
                'speed': track['speed'],
                'angle': track['angle'],
                'world_position': track['world_position']
            }
            
            self.vehicle_history[track_id].append(state)
            
            # Keep only recent history (last 30 frames)
            if len(self.vehicle_history[track_id]) > 30:
                self.vehicle_history[track_id] = self.vehicle_history[track_id][-30:]
    
    def get_vehicle_trajectory(self, track_id, num_points=10):
        """Get recent trajectory points for a vehicle."""
        if track_id not in self.vehicle_history:
            return []
        
        history = self.vehicle_history[track_id]
        return history[-num_points:] if len(history) >= num_points else history
    
    def predict_vehicle_position(self, track_id, time_ahead=1.0):
        """Predict vehicle position based on current motion."""
        if track_id not in self.vehicle_history:
            return None
        
        history = self.vehicle_history[track_id]
        if len(history) < 2:
            return None
        
        current_state = history[-1]
        
        if current_state['world_position']:
            x, y = current_state['world_position']
            speed = current_state['speed']
            angle_rad = math.radians(current_state['angle'])
            
            # Simple linear prediction
            pred_x = x + speed * time_ahead * math.cos(angle_rad)
            pred_y = y + speed * time_ahead * math.sin(angle_rad)
            
            return {'x': pred_x, 'y': pred_y, 'confidence': 0.8}
        
        return None
    
    def get_v2x_relevant_tracks(self, ego_track_id=None, max_distance=100.0):
        """Get tracks relevant for V2X communication."""
        if not ego_track_id:
            return list(self.vehicle_history.keys())
        
        # Filter tracks within V2X communication range
        relevant_tracks = []
        ego_history = self.vehicle_history.get(ego_track_id)
        
        if not ego_history:
            return list(self.vehicle_history.keys())
        
        ego_pos = ego_history[-1]['world_position']
        if not ego_pos:
            return list(self.vehicle_history.keys())
        
        for track_id, history in self.vehicle_history.items():
            if track_id == ego_track_id:
                continue
            
            if not history:
                continue
            
            track_pos = history[-1]['world_position']
            if not track_pos:
                continue
            
            # Calculate distance
            distance = math.sqrt((ego_pos[0] - track_pos[0])**2 + 
                               (ego_pos[1] - track_pos[1])**2)
            
            if distance <= max_distance:
                relevant_tracks.append(track_id)
        
        return relevant_tracks