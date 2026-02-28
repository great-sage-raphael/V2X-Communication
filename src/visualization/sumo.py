import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import traci

# --- CONFIGURATION ---
sumoBinary = "/Users/vinayakprakash/sumo/bin/sumo"
sumoConfig = "/Users/vinayakprakash/Documents/V2X-COMMUNICATION/src/simulation/sumo_data/map.sumocfg"

  # Point this to your actual config file
COMM_RADIUS = 100  # Meters
MAX_VEHICLES = 50 # Limit for the RL Agent's observation space

# --- SETUP SUMO ---
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

sumoCmd = [sumoBinary, "-c", sumoConfig]
traci.start(sumoCmd)

# --- MATPLOTLIB SETUP ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(1600, 2600)  # Adjust to your map size
ax.set_ylim(1600, 2600)
ax.set_title(f"Digital Twin: V2X Links (Radius: {COMM_RADIUS}m)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

# 1. Vehicle Dots
vehicle_scat = ax.scatter([], [], c='blue', s=50, zorder=3)

# 2. Connection Lines (Optimized with LineCollection)
# We create one empty collection and just update data later
link_collection = LineCollection([], colors='red', linewidths=0.5, alpha=0.5, zorder=1)
ax.add_collection(link_collection)

# 3. Text Labels
vehicle_texts = []  # We will reuse text objects to speed up rendering

def get_global_state():
    """
    Extracts the State Matrix for the RL Agent.
    Returns: Numpy Array [MAX_VEHICLES, 4] -> (x, y, speed, waiting_time)
    """
    vehicle_ids = traci.vehicle.getIDList()
    state_matrix = np.zeros((MAX_VEHICLES, 4), dtype=np.float32)
    
    active_nodes = {} # Store for visualization
    
    for i, vid in enumerate(vehicle_ids):
        if i >= MAX_VEHICLES: break
        
        # Data Extraction
        x, y = traci.vehicle.getPosition(vid)
        speed = traci.vehicle.getSpeed(vid)
        
        # Populate State Matrix (Normalized inputs are better for RL)
        state_matrix[i] = [x, y, speed, 0.0] 
        
        # Store for Viz
        active_nodes[vid] = (x, y)

    return state_matrix, active_nodes, vehicle_ids

# --- MAIN LOOP ---
try:
    for step in range(2000):
        traci.simulationStep()
        
        # 1. Get Data from SUMO
        state, nodes, v_ids = get_global_state()
        
        # 2. Calculate Connections (The Logic)
        lines = []
        positions = []
        
        # Extract just coordinates for faster distance calc
        coords = np.array(list(nodes.values()))
        
        if len(coords) > 1:
            positions = coords
            
            # Simple N^2 distance check (Fast enough for <100 cars)
            # For 1000+ cars, we would use a KDTree
            for i in range(len(v_ids)):
                for j in range(i + 1, len(v_ids)):
                    p1 = nodes[v_ids[i]]
                    p2 = nodes[v_ids[j]]
                    dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    
                    if dist <= COMM_RADIUS:
                        lines.append([p1, p2])
        
        # 3. Update Visualization (The Fast Way)
        
        # Update Dots
        if len(positions) > 0:
            vehicle_scat.set_offsets(positions)
        else:
            vehicle_scat.set_offsets(np.zeros((0, 2))) # Clear if empty
            
        # Update Lines
        link_collection.set_segments(lines)
        
        # Update Text (Reuse objects to avoid lag)
        # Ensure we have enough text objects
        while len(vehicle_texts) < len(v_ids):
            t = ax.text(0, 0, "", fontsize=8)
            vehicle_texts.append(t)
            
        # Update text positions and strings
        for i, txt in enumerate(vehicle_texts):
            if i < len(v_ids):
                vid = v_ids[i]
                x, y = nodes[vid]
                txt.set_position((x, y))
                txt.set_text(vid)
                txt.set_visible(True)
            else:
                txt.set_visible(False) # Hide unused labels

        # Pause briefly to let plot render
        fig.canvas.draw_idle()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Simulation stopped by user.")

traci.close()
plt.ioff()
plt.show()