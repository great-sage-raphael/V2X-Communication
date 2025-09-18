# src/perception/sumo_interface.py
"""
SUMO interface using TraCI.
Provides SumoRunner which can either:
 - start SUMO subprocess from a .sumocfg config, or
 - connect to an already-running SUMO (via traci.connect).

Usage:
    runner = SumoRunner(sumocfg="path/to/sim.sumocfg", step_length=0.1)
    for timestep, vehicles in runner.run():
        # vehicles -> list of dicts: {"id":..., "x":..., "y":..., "speed":..., "angle":...}
        ...
"""

import traci
import traci.constants as tc
import subprocess
import os

class SumoRunner:
    def __init__(self, sumocfg=None, sumo_binary="sumo", step_length=0.1, use_gui=False):
        """
        sumocfg: path to .sumocfg file (if starting SUMO). If None, assumes SUMO is already running.
        sumo_binary: "sumo" or "sumo-gui" (depends if you want GUI)
        step_length: simulation step length in seconds
        use_gui: whether to launch sumo-gui (True) or CLI (False)
        """
        self.sumocfg = sumocfg
        self.sumo_binary = "sumo-gui" if use_gui else sumo_binary
        self.step_length = step_length
        self.sumo_process = None
        self.connected = False

    def start(self):
        """Start SUMO process (if sumocfg provided) and connect TraCI."""
        if self.sumocfg:
            # Launch SUMO with TraCI remote port (default 8813). Use --start to avoid waiting.
            cmd = [self.sumo_binary, "-c", self.sumocfg, "--step-length", str(self.step_length), "--start"]
            # Use XML output for FCD if needed by adding --fcd-output but here we use traci directly.
            self.sumo_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # connect to traci default port 8813; if SUMO already started with remote-port X, adjust here.
        traci.init(8813)
        self.connected = True

    def run(self):
        """
        Generator: yields (timestep, vehicles_list) until simulation ends.
        Each vehicles_list is list of dict: {"id":id, "x":x, "y":y, "speed":speed, "angle":angle}
        """
        if not self.connected:
            self.start()

        step = 0
        try:
            while True:
                # Run one simulation step
                traci.simulationStep()
                # Get list of vehicle IDs currently in simulation
                veh_ids = traci.vehicle.getIDList()
                vehicles = []
                for vid in veh_ids:
                    # x,y are SUMO world coordinates (meters)
                    x, y = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    angle = traci.vehicle.getAngle(vid)
                    vehicles.append({"id": vid, "x": float(x), "y": float(y), "speed": float(speed), "angle": float(angle)})
                yield step, vehicles
                step += 1
                # stop condition: if simulation end reached, traci will raise exception or vehicle count zero.
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break
        finally:
            # cleanup
            traci.close()
            if self.sumo_process:
                self.sumo_process.terminate()
