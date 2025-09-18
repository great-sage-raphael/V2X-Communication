# src/perception/detection_module.py
"""
SUMO-based detection module that works directly with SUMO vehicle data
instead of running YOLO on video frames.
"""

import numpy as np
import traci

class SumoDetectionModule:
    def __init__(self, vehicle_types=None):
        """
        Initialize SUMO detection module.
        
        Args:
            vehicle_types: Dict mapping SUMO vehicle types to class IDs
                          e.g., {"passenger": 1, "truck": 2, "bus": 3}
        """
        self.vehicle_types = vehicle_types or {
            "passenger": 1,
            "truck": 2, 
            "bus": 3,
            "motorcycle": 4,
            "bicycle": 5,
            "pedestrian": 0
        }
        
        # Default class mapping
        self.class_names = {
            0: "person",
            1: "car", 
            2: "truck",
            3: "bus",
            4: "motorcycle",
            5: "bicycle"
        }
    
    def detect_from_sumo_data(self, vehicles, converter):
        """
        Create detections directly from SUMO vehicle data.
        
        Args:
            vehicles: List of vehicle dicts from SUMO
            converter: SumoToDetections instance for coordinate conversion
            
        Returns:
            np.array: Detection array [x1, y1, x2, y2, conf, cls]
        """
        detections = []
        
        for vehicle in vehicles:
            # Get vehicle type and map to class
            try:
                # Try to get vehicle type from TraCI if connected
                veh_type = traci.vehicle.getVehicleClass(vehicle['id'])
                class_id = self.vehicle_types.get(veh_type, 1)  # Default to car
            except:
                # Fallback: infer from vehicle ID or use default
                class_id = self._infer_class_from_id(vehicle['id'])
            
            # Convert world coordinates to pixel coordinates
            px, py = converter.world_to_pixel(vehicle['x'], vehicle['y'])
            
            # Create bounding box based on vehicle type and speed
            bbox_size = self._get_bbox_size(class_id, vehicle.get('speed', 0))
            half_w, half_h = bbox_size
            
            x1 = px - half_w
            y1 = py - half_h  
            x2 = px + half_w
            y2 = py + half_h
            
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(converter.image_w - 1, x2)
            y2 = min(converter.image_h - 1, y2)
            
            # Confidence is always 1.0 for SUMO data (perfect detection)
            conf = 1.0
            
            detections.append([x1, y1, x2, y2, conf, class_id])
        
        return np.array(detections, dtype=float)
    
    def _infer_class_from_id(self, vehicle_id):
        """Infer vehicle class from ID string."""
        vehicle_id = vehicle_id.lower()
        
        if 'truck' in vehicle_id or 'freight' in vehicle_id:
            return 2  # truck
        elif 'bus' in vehicle_id:
            return 3  # bus
        elif 'bike' in vehicle_id or 'motorcycle' in vehicle_id:
            return 4  # motorcycle
        elif 'bicycle' in vehicle_id or 'cycle' in vehicle_id:
            return 5  # bicycle
        elif 'pedestrian' in vehicle_id or 'person' in vehicle_id:
            return 0  # person
        else:
            return 1  # default to car
    
    def _get_bbox_size(self, class_id, speed):
        """Get bounding box size based on vehicle class and speed."""
        # Base sizes (half width, half height) in pixels
        base_sizes = {
            0: (8, 8),   # person
            1: (12, 6),  # car
            2: (18, 8),  # truck
            3: (20, 10), # bus
            4: (6, 4),   # motorcycle
            5: (4, 4)    # bicycle
        }
        
        base_w, base_h = base_sizes.get(class_id, (12, 6))
        
        # Slightly increase size for faster vehicles (motion blur effect)
        speed_factor = 1.0 + min(speed * 0.01, 0.3)  # Max 30% increase
        
        return int(base_w * speed_factor), int(base_h * speed_factor)
    
    def detect(self, frame_data):
        """
        Main detection method compatible with existing pipeline.
        
        Args:
            frame_data: Dict containing 'vehicles' and other SUMO data
            
        Returns:
            np.array: Detection array [x1, y1, x2, y2, conf, cls]
        """
        if 'detections' in frame_data:
            # Already converted detections
            return frame_data['detections']
        elif 'vehicles' in frame_data:
            # Need to convert from vehicles (requires converter)
            # This assumes converter is passed or available
            raise ValueError("Detection from raw vehicles requires SumoToDetections converter")
        else:
            return np.array([])
    
    def get_class_names(self):
        """Get class names mapping."""
        return self.class_names