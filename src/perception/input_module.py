# src/perception/input_module.py
"""
SUMO-based input module to replace video capture.
Provides vehicle data from SUMO simulation instead of video frames.
"""

import numpy as np
from simulation.sumo_interface import SumoRunner
from simulation.sumo_to_detection import SumoToDetections

class SumoInputModule:
    def __init__(self, sumocfg=None, sumo_binary="sumo", step_length=0.1, 
                 use_gui=False, world_bbox=None, image_size=(800, 800)):
        """
        Initialize SUMO input module.
        
        Args:
            sumocfg: Path to SUMO configuration file
            sumo_binary: SUMO binary to use ("sumo" or "sumo-gui")
            step_length: Simulation step length in seconds
            use_gui: Whether to use GUI
            world_bbox: World bounding box (xmin, ymin, xmax, ymax) in meters
            image_size: Output image size for visualization (width, height)
        """
        self.sumo_runner = SumoRunner(
            sumocfg=sumocfg,
            sumo_binary=sumo_binary,
            step_length=step_length,
            use_gui=use_gui
        )
        self.converter = SumoToDetections(
            world_bbox=world_bbox,
            image_size=image_size
        )
        self.simulation_generator = None
        self.current_vehicles = []
        self.current_timestep = 0
        self.is_running = False
        
    def start(self):
        """Start the SUMO simulation."""
        if not self.is_running:
            self.simulation_generator = self.sumo_runner.run()
            self.is_running = True
    
    def get_frame_data(self):
        """
        Get current frame data from SUMO simulation.
        Returns:
            dict: Contains 'detections', 'vehicles', 'timestep', and optional 'frame'
        """
        if not self.is_running:
            self.start()
        
        try:
            self.current_timestep, self.current_vehicles = next(self.simulation_generator)
            
            # Convert vehicles to detection format
            detections = self.converter.vehicles_to_detections(self.current_vehicles)
            
            # Create optional visualization frame
            frame = self.create_visualization_frame()
            
            return {
                'detections': detections,
                'vehicles': self.current_vehicles,
                'timestep': self.current_timestep,
                'frame': frame,
                'success': True
            }
            
        except StopIteration:
            # Simulation ended
            self.is_running = False
            return {
                'detections': np.array([]),
                'vehicles': [],
                'timestep': self.current_timestep,
                'frame': None,
                'success': False
            }
    
    def create_visualization_frame(self):
        """
        Create a top-down visualization frame from vehicle positions.
        Returns:
            numpy.ndarray: RGB image array for visualization
        """
        import cv2
        
        # Create blank frame
        frame = np.zeros((self.converter.image_h, self.converter.image_w, 3), dtype=np.uint8)
        
        # Draw vehicles as circles/rectangles
        for vehicle in self.current_vehicles:
            px, py = self.converter.world_to_pixel(vehicle['x'], vehicle['y'])
            
            # Draw vehicle as a small rectangle
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            
            # Draw vehicle ID
            cv2.putText(frame, str(vehicle['id']), (px + 7, py - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw direction arrow based on angle
            angle_rad = np.radians(vehicle['angle'])
            end_x = px + int(15 * np.cos(angle_rad))
            end_y = py + int(15 * np.sin(angle_rad))
            cv2.arrowedLine(frame, (px, py), (end_x, end_y), (0, 0, 255), 2)
        
        return frame
    
    def get_vehicles(self):
        """Get current vehicle list."""
        return self.current_vehicles
    
    def get_timestep(self):
        """Get current simulation timestep."""
        return self.current_timestep
    
    def release(self):
        """Clean up resources."""
        if self.is_running:
            try:
                # Close the generator
                if self.simulation_generator:
                    self.simulation_generator.close()
            except:
                pass
        self.is_running = False