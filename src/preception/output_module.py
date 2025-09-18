# src/perception/sumo_output_module.py
"""
Enhanced output module for SUMO-based V2X communication system.
Handles both visualization and V2X message generation.
"""

import cv2
import json
import numpy as np
from datetime import datetime

class SumoOutputModule:
    def __init__(self, class_names=None):
        """
        Initialize SUMO output module.
        
        Args:
            class_names: Dict mapping class IDs to names
        """
        self.class_names = class_names if class_names else {
            0: "person",
            1: "car", 
            2: "truck",
            3: "bus",
            4: "motorcycle",
            5: "bicycle"
        }
        
        # Colors for different vehicle types (BGR format for OpenCV)
        self.colors = {
            0: (0, 255, 255),    # person - yellow
            1: (0, 255, 0),      # car - green
            2: (255, 0, 0),      # truck - blue
            3: (0, 0, 255),      # bus - red
            4: (255, 0, 255),    # motorcycle - magenta
            5: (255, 255, 0)     # bicycle - cyan
        }
    
    def draw(self, frame, tracks, show_speed=True, show_trajectory=False):
        """
        Draw tracked vehicles on frame with enhanced V2X information.
        
        Args:
            frame: Image frame to draw on
            tracks: List of tracked objects
            show_speed: Whether to show speed information
            show_trajectory: Whether to show vehicle trajectory
            
        Returns:
            frame: Frame with drawn annotations
        """
        if frame is None:
            return None
            
        for track in tracks:
            # Extract track information
            if hasattr(track, 'tlbr'):
                # ByteTracker format
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id
                class_id = getattr(track, 'cls', 1)
                conf = getattr(track, 'score', 1.0)
            else:
                # Dict format
                x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
                track_id = track.get('track_id', 0)
                class_id = track.get('class', 1)
                conf = track.get('score', 1.0)
            
            # Get color for this vehicle type
            color = self.colors.get(class_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            class_name = self.class_names.get(class_id, "unknown")
            label = f"ID:{track_id} {class_name}"
            
            # Add confidence if available
            if conf < 1.0:
                label += f" {conf:.2f}"
            
            # Add speed information if available
            if show_speed and hasattr(track, 'speed'):
                label += f" {track.speed:.1f}m/s"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw direction arrow if angle information is available
            if hasattr(track, 'angle'):
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                angle_rad = np.radians(track.angle)
                
                arrow_length = 30
                end_x = center_x + int(arrow_length * np.cos(angle_rad))
                end_y = center_y + int(arrow_length * np.sin(angle_rad))
                
                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                               (0, 0, 255), 2, tipLength=0.3)
        
        # Add simulation info
        self._draw_sim_info(frame)
        
        return frame
    
    def _draw_sim_info(self, frame):
        """Draw simulation information on frame."""
        if frame is None:
            return
            
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add mode indicator
        cv2.putText(frame, "SUMO V2X Mode", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def to_json(self, tracks, timestep=0, include_v2x_data=True):
        """
        Convert tracks to JSON format for V2X communication.
        
        Args:
            tracks: List of tracked objects
            timestep: Current simulation timestep
            include_v2x_data: Whether to include V2X-specific data
            
        Returns:
            dict: JSON-serializable data structure
        """
        vehicles = []
        
        for track in tracks:
            # Extract track information
            if hasattr(track, 'tlbr'):
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id
                class_id = getattr(track, 'cls', 1)
                conf = getattr(track, 'score', 1.0)
            else:
                x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
                track_id = track.get('track_id', 0)
                class_id = track.get('class', 1)
                conf = track.get('score', 1.0)
            
            vehicle_data = {
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "class": self.class_names.get(class_id, "unknown"),
                "class_id": class_id,
                "confidence": conf,
                "timestamp": timestep
            }
            
            # Add V2X-specific data if available
            if include_v2x_data:
                # Position data
                if hasattr(track, 'world_pos'):
                    vehicle_data["position"] = {
                        "x": track.world_pos[0],
                        "y": track.world_pos[1]
                    }
                
                # Motion data
                if hasattr(track, 'speed'):
                    vehicle_data["speed"] = track.speed
                
                if hasattr(track, 'angle'):
                    vehicle_data["heading"] = track.angle
                
                # V2X message type classification
                vehicle_data["v2x_type"] = self._classify_v2x_message_type(class_id, 
                                                                        getattr(track, 'speed', 0))
            
            vehicles.append(vehicle_data)
        
        # Create complete message
        message = {
            "timestamp": timestep,
            "message_type": "CAM",  # Cooperative Awareness Message
            "vehicles": vehicles,
            "total_vehicles": len(vehicles)
        }
        
        return message
    
    def _classify_v2x_message_type(self, class_id, speed):
        """Classify V2X message type based on vehicle characteristics."""
        if class_id == 0:  # pedestrian
            return "VRU"  # Vulnerable Road User
        elif speed < 0.5:  # stationary
            return "CAM_STATIONARY"
        elif speed > 15:  # high speed
            return "CAM_HIGH_SPEED"
        else:
            return "CAM_NORMAL"
    
    def save_v2x_message(self, message, filepath):
        """Save V2X message to file."""
        with open(filepath, 'w') as f:
            json.dump(message, f, indent=2)
    
    def create_v2x_broadcast(self, tracks, ego_vehicle_id=None):
        """
        Create V2X broadcast message from ego vehicle perspective.
        
        Args:
            tracks: List of tracked objects
            ego_vehicle_id: ID of ego vehicle (if None, creates general broadcast)
            
        Returns:
            dict: V2X broadcast message
        """
        message = self.to_json(tracks, include_v2x_data=True)
        
        # Add broadcast-specific information
        message["message_type"] = "V2X_BROADCAST"
        message["ego_vehicle"] = ego_vehicle_id
        message["range_km"] = 1.0  # V2X communication range
        
        # Filter nearby vehicles (within V2X range)
        if ego_vehicle_id:
            # This would require distance calculation in real implementation
            message["nearby_vehicles"] = [v for v in message["vehicles"] 
                                        if v["id"] != ego_vehicle_id]
        
        return message