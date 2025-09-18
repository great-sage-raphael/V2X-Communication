# main_v2x_sumo.py
"""
Complete example showing how to integrate all SUMO-based modules
to replace the video-capture based pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

# Import your fixed modules
from perception.input_module import SumoInputModule
from perception.detection_module import SumoDetectionModule
from perception.tracking_module import EnhancedTracker
from perception.output_module import SumoOutputModule
from simulation.sumo_interface import SumoRunner

def create_v2x_config():
    """Create default V2X configuration."""
    return {
        # SUMO Configuration
        'sumocfg': 'scenarios/quickstart.sumocfg',  # Your SUMO scenario
        'sumo_binary': 'sumo-gui',  # Use GUI for visualization
        'step_length': 0.1,  # 100ms simulation steps
        'use_gui': True,
        
        # World mapping
        'world_bbox': None,  # Auto-compute or set to (xmin, ymin, xmax, ymax)
        'image_size': (1200, 800),  # Output resolution
        'default_bbox_size_m': 2.0,  # Vehicle size in meters
        
        # Vehicle type mapping
        'vehicle_types': {
            'passenger': 1,
            'truck': 2,
            'bus': 3,
            'emergency': 4,
            'motorcycle': 5,
            'bicycle': 6
        },
        
        # Tracking configuration
        'track_thresh': 0.6,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'use_sumo_matching': True,
        
        # V2X Configuration
        'ego_vehicle_id': 'veh_0',  # Your ego vehicle
        'v2x_range_m': 300.0,  # V2X communication range
        'broadcast_interval': 5,  # Broadcast every N frames
        
        # Output configuration
        'save_output': True,
        'output_dir': './v2x_results/',
        'show_visualization': True,
        'save_frames': False,
        'save_v2x_messages': True
    }

def run_v2x_pipeline(config):
    """Run the complete V2X pipeline with SUMO."""
    print("🚗 Starting V2X SUMO Pipeline...")
    
    # Initialize components
    print("📡 Initializing modules...")
    
    # Input: SUMO simulation data
    input_module = SumoInputModule(
        sumocfg=config['sumocfg'],
        sumo_binary=config['sumo_binary'],
        step_length=config['step_length'],
        use_gui=config['use_gui'],
        world_bbox=config['world_bbox'],
        image_size=config['image_size']
    )
    
    # Detection: Convert SUMO data to detection format
    detection_module = SumoDetectionModule(
        vehicle_types=config['vehicle_types']
    )
    
    # Tracking: Enhanced tracking with SUMO integration
    tracker = EnhancedTracker(
        frame_rate=int(1.0 / config['step_length']),
        track_thresh=config['track_thresh'],
        track_buffer=config['track_buffer'],
        match_thresh=config['match_thresh'],
        use_sumo_matching=config['use_sumo_matching']
    )
    
    # Output: V2X message generation and visualization
    output_module = SumoOutputModule(
        class_names=detection_module.get_class_names()
    )
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("🎯 Pipeline initialized successfully!")
    print("🎮 Controls: 'q' to quit, 'p' to pause, 's' to save state")
    print("=" * 60)
    
    frame_count = 0
    v2x_messages = []
    
    try:
        input_module.start()
        
        while True:
            # 1. Get SUMO simulation data
            frame_data = input_module.get_frame_data()
            
            if not frame_data['success']:
                print("✅ Simulation completed!")
                break
            
            timestep = frame_data['timestep']
            vehicles = frame_data['vehicles']
            detections = frame_data['detections']
            frame = frame_data.get('frame')
            
            print(f"📊 Frame {frame_count}: Step {timestep:.1f}s, {len(vehicles)} vehicles")
            
            # 2. Enhanced tracking with SUMO data
            tracks = tracker.update(detections, 
                                  frame if frame is not None else np.zeros((800, 800, 3)),
                                  sumo_vehicles=vehicles)
            
            # 3. Generate V2X messages
            if frame_count % config['broadcast_interval'] == 0:
                v2x_message = output_module.create_v2x_broadcast(
                    tracks, 
                    ego_vehicle_id=config['ego_vehicle_id']
                )
                v2x_messages.append(v2x_message)
                
                # Print V2X summary
                nearby_vehicles = len([t for t in tracks if t['track_id'] != config.get('ego_vehicle_id')])
                print(f"📡 V2X Broadcast: {nearby_vehicles} nearby vehicles detected")
            
            # 4. Visualization and output
            if config['show_visualization'] and frame is not None:
                display_frame = output_module.draw(
                    frame.copy(), 
                    tracks,
                    show_speed=True,
                    show_trajectory=False
                )
                
                # Add pipeline information
                import cv2
                info_text = f"Frame: {frame_count} | Timestep: {timestep:.1f}s | Vehicles: {len(tracks)}"
                cv2.putText(display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_frame, "V2X SUMO Pipeline", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.imshow("V2X SUMO Perception", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Stopping pipeline...")
                    break
                elif key == ord('p'):
                    print("⏸️  Paused - Press any key to continue")
                    cv2.waitKey(0)
                elif key == ord('s'):
                    save_current_results(tracks, v2x_messages, frame_count, config['output_dir'])
            
            frame_count += 1
            
            # Optional: limit frames for testing
            if frame_count >= 1000:  # Stop after 1000 frames
                print("🔄 Reached frame limit")
                break
                
    except KeyboardInterrupt:
        print("⚠️  Pipeline interrupted by user")
    
    finally:
        # Cleanup and save results
        print("💾 Saving results...")
        
        if config['save_v2x_messages'] and v2x_messages:
            save_v2x_messages(v2x_messages, config['output_dir'])
        
        input_module.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Pipeline completed!")
        print(f"📊 Processed {frame_count} frames")
        print(f"📡 Generated {len(v2x_messages)} V2X messages")
        print(f"💾 Results saved to: {config['output_dir']}")

def save_current_results(tracks, v2x_messages, frame_count, output_dir):
    """Save current pipeline results."""
    import json
    
    # Save current tracks
    tracks_data = []
    for track in tracks:
        # Convert numpy types to Python types for JSON serialization
        track_data = {}
        for k, v in track.items():
            if isinstance(v, np.ndarray):
                track_data[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                track_data[k] = v.item()
            else:
                track_data[k] = v
        tracks_data.append(track_data)
    
    with open(f"{output_dir}/tracks_frame_{frame_count}.json", 'w') as f:
        json.dump(tracks_data, f, indent=2)
    
    print(f"💾 Saved current state at frame {frame_count}")

def save_v2x_messages(v2x_messages, output_dir):
    """Save all V2X messages to file."""
    import json
    
    output_file = f"{output_dir}/v2x_messages_complete.json"
    with open(output_file, 'w') as f:
        json.dump(v2x_messages, f, indent=2)
    
    # Also save summary statistics
    summary = {
        'total_messages': len(v2x_messages),
        'message_types': {},
        'avg_vehicles_per_message': 0,
        'total_unique_vehicles': set()
    }
    
    total_vehicles = 0
    for msg in v2x_messages:
        msg_type = msg.get('message_type', 'unknown')
        summary['message_types'][msg_type] = summary['message_types'].get(msg_type, 0) + 1
        
        vehicles_in_msg = len(msg.get('vehicles', []))
        total_vehicles += vehicles_in_msg
        
        for vehicle in msg.get('vehicles', []):
            summary['total_unique_vehicles'].add(vehicle.get('id', 'unknown'))
    
    summary['avg_vehicles_per_message'] = total_vehicles / len(v2x_messages) if v2x_messages else 0
    summary['total_unique_vehicles'] = len(summary['total_unique_vehicles'])
    
    with open(f"{output_dir}/v2x_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='V2X SUMO Pipeline')
    parser.add_argument('--config', type=str, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--output', type=str, default='./v2x_results/', help='Output directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_v2x_config()
    
    # Override with command line arguments
    if args.config:
        config['sumocfg'] = args.config
    if args.gui:
        config['sumo_binary'] = 'sumo-gui'
        config['use_gui'] = True
    if args.no_viz:
        config['show_visualization'] = False
    if args.output:
        config['output_dir'] = args.output
    
    # Check if SUMO config exists
    if not os.path.exists(config['sumocfg']):
        print(f"❌ Error: SUMO config file not found: {config['sumocfg']}")
        print("Please provide a valid SUMO configuration file.")
        return
    
    # Run the pipeline
    try:
        import cv2
        import numpy as np
        run_v2x_pipeline(config)
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages: opencv-python, numpy")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()