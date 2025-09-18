# src/perception/sumo_pipeline.py
"""
Complete SUMO-based V2X communication pipeline.
Replaces video-based pipeline with SUMO simulation data.
"""

import cv2
import numpy as np
import time
import json
from sumo_input_module import SumoInputModule
from sumo_detection_module import SumoDetectionModule
from tracking_module import Tracker
from sumo_output_module import SumoOutputModule

class SumoV2XPipeline:
    def __init__(self, config):
        """
        Initialize SUMO V2X Pipeline.
        
        Args:
            config: Configuration dict with SUMO and pipeline parameters
        """
        self.config = config
        
        # Initialize modules
        self.input_module = SumoInputModule(
            sumocfg=config.get('sumocfg'),
            sumo_binary=config.get('sumo_binary', 'sumo'),
            step_length=config.get('step_length', 0.1),
            use_gui=config.get('use_gui', False),
            world_bbox=config.get('world_bbox'),
            image_size=config.get('image_size', (800, 800))
        )
        
        self.detection_module = SumoDetectionModule(
            vehicle_types=config.get('vehicle_types')
        )
        
        self.tracker = Tracker(
            frame_rate=int(1.0 / config.get('step_length', 0.1))
        )
        
        self.output_module = SumoOutputModule(
            class_names=self.detection_module.get_class_names()
        )
        
        # Pipeline state
        self.is_running = False
        self.frame_count = 0
        self.v2x_messages = []
        
        # Output settings
        self.save_output = config.get('save_output', False)
        self.output_dir = config.get('output_dir', './output/')
        self.show_visualization = config.get('show_visualization', True)
        
        # V2X settings
        self.ego_vehicle_id = config.get('ego_vehicle_id')
        self.v2x_broadcast_interval = config.get('v2x_broadcast_interval', 10)  # frames
    
    def run(self, max_frames=None):
        """
        Run the complete V2X perception pipeline.
        
        Args:
            max_frames: Maximum number of frames to process (None for unlimited)
        """
        print("Starting SUMO V2X Pipeline...")
        
        self.is_running = True
        self.frame_count = 0
        
        try:
            while self.is_running:
                # Get data from SUMO
                frame_data = self.input_module.get_frame_data()
                
                if not frame_data['success']:
                    print("Simulation ended or no more data")
                    break
                
                # Process frame
                self.process_frame(frame_data)
                
                # Check termination conditions
                if max_frames and self.frame_count >= max_frames:
                    break
                
                # Handle user input for GUI
                if self.show_visualization:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):  # pause
                        cv2.waitKey(0)
                    elif key == ord('s'):  # save current state
                        self.save_current_state()
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("Pipeline interrupted by user")
        
        finally:
            self.cleanup()
    
    def process_frame(self, frame_data):
        """Process a single frame of SUMO data."""
        timestep = frame_data['timestep']
        vehicles = frame_data['vehicles']
        detections = frame_data['detections']
        frame = frame_data.get('frame')
        
        print(f"Frame {self.frame_count}: Timestep {timestep}, {len(vehicles)} vehicles")
        
        # Enhanced tracking with SUMO data
        tracks = self.tracker.update(detections, frame if frame is not None else np.zeros((800, 800, 3)))
        
        # Enhance tracks with SUMO vehicle data
        self.enhance_tracks_with_sumo_data(tracks, vehicles)
        
        # Generate V2X messages
        if self.frame_count % self.v2x_broadcast_interval == 0:
            v2x_message = self.generate_v2x_message(tracks, timestep)
            self.v2x_messages.append(v2x_message)
            
            if self.config.get('print_v2x_messages', False):
                print(f"V2X Message: {len(tracks)} vehicles detected")
        
        # Visualization
        if self.show_visualization and frame is not None:
            display_frame = self.output_module.draw(
                frame.copy(), 
                tracks,
                show_speed=True,
                show_trajectory=False
            )
            
            # Add frame information
            cv2.putText(display_frame, f"Frame: {self.frame_count} | Step: {timestep}", 
                       (10, display_frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("SUMO V2X Perception", display_frame)
        
        # Save output if requested
        if self.save_output:
            self.save_frame_output(tracks, timestep, frame)
    
    def enhance_tracks_with_sumo_data(self, tracks, vehicles):
        """Enhance tracking results with SUMO vehicle data."""
        # Create vehicle lookup by position proximity
        vehicle_lookup = {v['id']: v for v in vehicles}
        
        for track in tracks:
            # Try to match track with SUMO vehicle
            # This is a simplified matching - in practice you'd use more sophisticated methods
            if hasattr(track, 'track_id'):
                # For now, assume track_id corresponds to vehicle index
                if track.track_id < len(vehicles):
                    vehicle = vehicles[track.track_id % len(vehicles)]
                    
                    # Add SUMO data to track
                    track.world_pos = (vehicle['x'], vehicle['y'])
                    track.speed = vehicle['speed']
                    track.angle = vehicle['angle']
                    track.sumo_id = vehicle['id']
    
    def generate_v2x_message(self, tracks, timestep):
        """Generate V2X communication message."""
        if self.ego_vehicle_id:
            message = self.output_module.create_v2x_broadcast(tracks, self.ego_vehicle_id)
        else:
            message = self.output_module.to_json(tracks, timestep, include_v2x_data=True)
        
        message['pipeline_frame'] = self.frame_count
        return message
    
    def save_frame_output(self, tracks, timestep, frame):
        """Save frame output data."""
        import os
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Save tracking results
        tracking_data = self.output_module.to_json(tracks, timestep)
        with open(f"{self.output_dir}/tracks_{self.frame_count:06d}.json", 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        # Save frame if available
        if frame is not None and self.config.get('save_frames', False):
            annotated_frame = self.output_module.draw(frame.copy(), tracks)
            cv2.imwrite(f"{self.output_dir}/frame_{self.frame_count:06d}.jpg", annotated_frame)
    
    def save_current_state(self):
        """Save current pipeline state."""
        state = {
            'frame_count': self.frame_count,
            'config': self.config,
            'v2x_messages': self.v2x_messages[-10:],  # Last 10 messages
            'timestamp': time.time()
        }
        
        with open(f"{self.output_dir}/pipeline_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"State saved at frame {self.frame_count}")
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up pipeline...")
        
        self.is_running = False
        
        # Save final V2X messages
        if self.v2x_messages and self.save_output:
            import os
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            with open(f"{self.output_dir}/v2x_messages.json", 'w') as f:
                json.dump(self.v2x_messages, f, indent=2)
        
        # Close input module
        self.input_module.release()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print(f"Pipeline completed. Processed {self.frame_count} frames.")
        print(f"Generated {len(self.v2x_messages)} V2X messages.")

def main():
    """Example usage of the SUMO V2X pipeline."""
    # Example configuration
    config = {
        # SUMO settings
        'sumocfg': 'simulation/quickstart.sumocfg',  # Path to your SUMO config
        'sumo_binary': 'sumo',  # or 'sumo-gui' for visualization
        'step_length': 0.1,  # 100ms steps
        'use_gui': True,
        
        # Perception settings
        'world_bbox': None,  # Auto-compute from data
        'image_size': (1200, 800),
        'vehicle_types': {
            'passenger': 1,
            'truck': 2,
            'bus': 3,
            'emergency': 4
        },
        
        # Pipeline settings
        'save_output': True,
        'output_dir': './v2x_output/',
        'show_visualization': True,
        'save_frames': True,
        
        # V2X settings
        'ego_vehicle_id': 'ego_0',  # Your ego vehicle ID
        'v2x_broadcast_interval': 5,  # Broadcast every 5 frames
        'print_v2x_messages': True
    }
    
    # Create and run pipeline
    pipeline = SumoV2XPipeline(config)
    
    try:
        pipeline.run(max_frames=1000)  # Run for 1000 frames max
    except Exception as e:
        print(f"Pipeline error: {e}")
        pipeline.cleanup()

if __name__ == "__main__":
    main()