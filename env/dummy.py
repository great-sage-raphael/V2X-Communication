import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import os
import math

MAX_VEHICLES = 20 

sumoBinary = "/Users/vinayakprakash/sumo/bin/sumo"
sumoConfig = "/Users/vinayakprakash/Documents/V2X-COMMUNICATION/src/simulation/sumo_data/map.sumocfg"

class V2XEnv(gym.Env):

    def _init_(self):
        super(V2XEnv,self)._init_()

           # --- 1. ACTION SPACE (The Choices) ---

        # We want to choose Mode 0 (V2I) or Mode 1 (V2V) for EACH car.

        # Format: MultiDiscrete([2, 2, 2, ...]) -> A vector of 0s and 1s.

        # Note: This works natively with PPO. For DQN, we might need a wrapper later, 

        # but this is the most accurate physical representation.

        self.action_space = spaces.MultiDiscrete([2] * MAX_VEHICLES)



        # --- 2. OBSERVATION SPACE (The "Digital Twin" Matrix) ---

        # Matrix: [MAX_VEHICLES, 4] 

        # Columns: [Position X, Position Y, Speed, Neighbor_Count]

        self.observation_space = spaces.Box(

            low=-np.inf, high=np.inf, shape=(MAX_VEHICLES, 4), dtype=np.float32

        )
        self.step_count = 0
        self.vehicle_list = [] # Keep track of tracked vehicles

    def reset(self, seed=None, options=None):

        """Restarts the simulation."""

        super().reset(seed=seed)

        try:

            traci.close()

        except:

            pass



        # Start SUMO

        sumo_cmd = [sumoBinary, "-c", sumoConfig]

        traci.start(sumo_cmd)

        

        self.step_count = 0

        return self._get_state(), {}



    def step(self, action):

        """

        The Main Loop:

        1. Agent chooses modes (V2I/V2V).

        2. Environment calculates packet success based on Physics.

        3. SUMO updates positions.

        4. Reward is calculated.

        """

        # 1. APPLY ACTIONS (Simulate Network Physics)
        # We calculate how many packets were delivered successfully based on the mode chosen.
        successful_packets = 0
        total_attempts = 0
        current_ids = traci.vehicle.getIDList()
        # Loop through every active car and check its assigned mode
        for i, vid in enumerate(current_ids):

            if i >= MAX_VEHICLES: break
            chosen_mode = action[i] # 0 = V2I, 1 = V2V

            if chosen_mode == 0: # V2I (Base Station)
                # Rule: V2I is reliable (99%) but has Latency Penalty
                # In simulation, we assume it always delivers, but we penalize the Reward slightly for delay.
                successful_packets += 1 
            elif chosen_mode == 1: # V2V (Direct)
                     # Rule: V2V is instant, BUT fails if too many neighbors (Interference).
                neighbors = self._count_neighbors(vid)
                       # PHYSICS MODEL: If > 5 neighbors, packet drop probability increases
                if neighbors > 5:
                    success = False  # Interference! Packet Lost.
                else:
                    success = True   # Clear channel.
                if success:
                    successful_packets += 1
            total_attempts += 1
        # 2. STEP SIMULATION
        traci.simulationStep()
        self.step_count += 1
        # 3. GET NEW STATE
        state = self._get_state()
        # 4. CALCULATE REWARD
        # Reward = (Packets Delivered) - (Collisions * Penalty)
        reward = self._calculate_reward(successful_packets, total_attempts)

        # 5. CHECK TERMINATION
        truncated = False
        if self.step_count >= 1000:

            truncated = True
        return state, reward, False, truncated, {}



    def _get_state(self):

        """Extracts the State Matrix from SUMO."""

        self.vehicle_list = traci.vehicle.getIDList()

        state_matrix = np.zeros((MAX_VEHICLES, 4), dtype=np.float32)

        

        for i, vid in enumerate(self.vehicle_list):

            if i >= MAX_VEHICLES: break

            

            x, y = traci.vehicle.getPosition(vid)

            speed = traci.vehicle.getSpeed(vid)

            neighbors = self._count_neighbors(vid)

            

            # Normalize for the AI (Speed/30, Pos/1000, Neighbors/10)

            state_matrix[i] = [x/1000.0, y/1000.0, speed/30.0, neighbors/10.0]

            

        return state_matrix



    def _count_neighbors(self, vid):

        """Returns number of cars within 50m (Interference Range)."""

        count = 0

        x1, y1 = traci.vehicle.getPosition(vid)

        

        for other_vid in self.vehicle_list:

            if vid == other_vid: continue

            

            x2, y2 = traci.vehicle.getPosition(other_vid)

            dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)

            

            if dist < 50.0: # 50m Interference Radius

                count += 1

        return count



    def _calculate_reward(self, successful_packets, total_attempts):

        """

        The Scoring System:

        - Big points for Delivery.

        - Huge penalty for Crashes.

        - Small penalty for V2I usage (to encourage V2V when safe).

        """

        collisions = traci.simulation.getCollidingVehiclesNumber()

        

        # 1. Throughput Reward

        # Avoid division by zero

        pdr = successful_packets / total_attempts if total_attempts > 0 else 0

        # 2. Crash Penalty

        crash_penalty = collisions * 10
        # Total

        return pdr - crash_penalty
        
    def close(self):
        traci.close()