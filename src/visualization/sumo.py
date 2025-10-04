import os
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import traci

# Set SUMO tools path
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# SUMO command
sumoBinary = "/Users/vinayakprakash/sumo/bin/sumo"
sumoConfig = "/Users/vinayakprakash/Documents/V2X-COMMUNICATION/src/simulation/sumo_data/map.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]

traci.start(sumoCmd)

# Plot setup
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(0, 1000)   # adjust according to your map size
ax.set_ylim(0, 1000)
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_title("Live V2X Digital Twin")

comm_radius = 100  # meters

# Keep track of scatter plots and lines
vehicle_scat = ax.scatter([], [], c='blue', s=50)
vehicle_texts = {}
edge_lines = []

for step in range(1000):
    traci.simulationStep()
    vehicle_ids = traci.vehicle.getIDList()
    
    nodes = {}
    edges = []
    
    # Get vehicle info
    for vid in vehicle_ids:
        x, y = traci.vehicle.getPosition(vid)
        speed = traci.vehicle.getSpeed(vid)
        angle = traci.vehicle.getAngle(vid)
        road = traci.vehicle.getRoadID(vid)
        lane = traci.vehicle.getLaneID(vid)
        accel = traci.vehicle.getAcceleration(vid)
        
        nodes[vid] = {
            "pos": (x, y),
            "speed": speed,
            "angle": angle,
            "road": road,
            "lane": lane,
            "accel": accel
        }
    
    # Determine communication links
    for i, v1 in enumerate(vehicle_ids):
        for v2 in vehicle_ids[i+1:]:
            dist = math.hypot(nodes[v1]['pos'][0] - nodes[v2]['pos'][0],
                              nodes[v1]['pos'][1] - nodes[v2]['pos'][1])
            if dist <= comm_radius:
                edges.append((v1, v2))
    
    # Update scatter positions
    positions = [nodes[vid]['pos'] for vid in vehicle_ids]
    vehicle_scat.set_offsets(positions)
    
    # Update or create text labels
    for vid, pos in zip(vehicle_ids, positions):
        if vid in vehicle_texts:
            vehicle_texts[vid].set_position(pos)
        else:
            vehicle_texts[vid] = ax.text(pos[0], pos[1], vid, fontsize=7)
    
    # Remove old edge lines
    for line in edge_lines:
        line.remove()
    edge_lines = []
    
    # Draw new edges
    for e in edges:
        x1, y1 = nodes[e[0]]['pos']
        x2, y2 = nodes[e[1]]['pos']
        line = mlines.Line2D([x1, x2], [y1, y2], color='red', alpha=0.3)
        ax.add_line(line)
        edge_lines.append(line)
    
    plt.pause(0.001)
    
traci.close()
plt.ioff()
plt.show()
