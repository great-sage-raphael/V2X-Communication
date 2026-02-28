"""
V2X Gymnasium Environment (v2x_env.py)
=======================================
Digital Twin environment wrapping the SUMO traffic simulator.

State Space : Box(MAX_VEHICLES, 4) → [x_norm, y_norm, speed_norm, neighbor_count_norm]
Action Space: MultiDiscrete([2] * MAX_VEHICLES) → 0 = V2I, 1 = V2V per vehicle

Reward Function (per step):
    reward = α·PDR − β·mean_latency_ms − γ·crash_penalty
    where α=1.0, β=0.02, γ=10.0

Network Physics Model:
    V2I: Reliable (99%), latency ~ N(20ms, 5ms)   — backhaul delay
    V2V: Instant (low latency ~ N(2ms, 0.5ms)), BUT fails if neighbors > 5 (interference)

Telemetry: Each episode accumulates per-step metrics. Call get_episode_summary() after
           the episode ends to retrieve a dict with averaged PDR, latency, and mode usage.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import math
import random

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
MAX_VEHICLES = 20                    # Size of the fixed observation matrix

# SUMO binaries — update paths if SUMO is installed elsewhere
SUMO_BINARY  = "/Users/vinayakprakash/sumo/bin/sumo"
SUMO_CONFIG  = "/Users/vinayakprakash/Documents/V2X-COMMUNICATION/src/simulation/sumo_data/map.sumocfg"

# Reward coefficients
ALPHA = 1.0    # PDR reward weight
BETA  = 0.02   # Latency penalty weight (per ms)
GAMMA = 10.0   # Collision penalty weight

# Network parameters
V2V_COMM_RADIUS          = 50.0   # metres — interference check radius
V2V_INTERFERENCE_THRESH  = 5      # neighbour count above which V2V fails
V2I_LATENCY_MEAN_MS      = 20.0   # V2I mean latency (backhaul)
V2I_LATENCY_STD_MS       = 5.0
V2V_LATENCY_MEAN_MS      = 2.0    # V2V mean latency (direct)
V2V_LATENCY_STD_MS       = 0.5
# ─────────────────────────────────────────────────────────────────────────────


class V2XEnv(gym.Env):
    """
    Intelligent V2X Communication Environment.

    Each step the agent selects a communication mode (V2I or V2V) for every
    active vehicle. The environment simulates packet delivery success and
    network latency based on a physics model, then advances the SUMO
    simulation by one timestep.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super(V2XEnv, self).__init__()

        # ── Action Space ──────────────────────────────────────────────────────
        # One binary choice per vehicle slot: 0 = V2I, 1 = V2V.
        # MultiDiscrete is natively supported by SB3's PPO and A2C.
        self.action_space = spaces.MultiDiscrete([2] * MAX_VEHICLES)

        # ── Observation Space ─────────────────────────────────────────────────
        # Matrix [MAX_VEHICLES, 4]: normalised (x, y, speed, neighbour_count)
        # Inactive vehicle slots remain zero-padded.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(MAX_VEHICLES, 4),
            dtype=np.float32
        )

        # Internal state
        self.step_count    = 0
        self.vehicle_list  = []

        # ── Per-episode telemetry (reset every episode) ───────────────────────
        self._ep_pdrs       = []   # PDR per step
        self._ep_latencies  = []   # mean latency (ms) per step
        self._ep_v2v_count  = 0    # total V2V actions taken this episode
        self._ep_v2i_count  = 0    # total V2I actions taken this episode

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Restart the SUMO simulation and clear episode telemetry."""
        super().reset(seed=seed)

        # Gracefully close any existing SUMO connection
        try:
            traci.close()
        except Exception:
            pass

        # Launch SUMO in headless (non-GUI) mode for RL training
        sumo_cmd = [SUMO_BINARY, "-c", SUMO_CONFIG, "--no-warnings"]
        traci.start(sumo_cmd)

        # Reset counters and telemetry buffers
        self.step_count    = 0
        self.vehicle_list  = []
        self._ep_pdrs      = []
        self._ep_latencies = []
        self._ep_v2v_count = 0
        self._ep_v2i_count = 0

        return self._get_state(), {}

    def step(self, action):
        """
        Execute one environment step.

        Pipeline:
            1. Apply agent's mode decisions → simulate packet delivery + latency.
            2. Advance SUMO by one simulation tick.
            3. Observe new state from the Digital Twin.
            4. Compute reward: PDR bonus − latency penalty − collision penalty.
            5. Check termination conditions.
        """

        # ── 1. APPLY ACTIONS: Simulate Network Physics ────────────────────────
        successful_packets = 0
        total_attempts     = 0
        step_latencies     = []  # latency (ms) for each attempted transmission

        current_ids = traci.vehicle.getIDList()

        for i, vid in enumerate(current_ids):
            if i >= MAX_VEHICLES:
                break

            # Synchronise the RL agent's decision with the current vehicle list.
            # 'action' is a fixed-length array; we index by position i.
            chosen_mode = int(action[i])  # 0 = V2I, 1 = V2V

            if chosen_mode == 0:
                # ── V2I (Base Station) ──
                # Reliable (simulated 99% success rate), but incurs backhaul delay.
                delivered   = random.random() < 0.99
                latency_ms  = max(1.0, np.random.normal(V2I_LATENCY_MEAN_MS, V2I_LATENCY_STD_MS))
                self._ep_v2i_count += 1

            else:
                # ── V2V (Direct Vehicle-to-Vehicle) ──
                # Ultra-low latency, but susceptible to channel congestion when
                # too many neighbours compete for the same frequency band.
                neighbors  = self._count_neighbors(vid)
                latency_ms = max(0.5, np.random.normal(V2V_LATENCY_MEAN_MS, V2V_LATENCY_STD_MS))
                # PHYSICS MODEL: packet drop probability rises beyond interference threshold
                delivered  = (neighbors <= V2V_INTERFERENCE_THRESH)
                self._ep_v2v_count += 1

            if delivered:
                successful_packets += 1

            step_latencies.append(latency_ms)
            total_attempts += 1

        # ── 2. ADVANCE SIMULATION ─────────────────────────────────────────────
        traci.simulationStep()
        self.step_count += 1

        # ── 3. NEW OBSERVATION ────────────────────────────────────────────────
        state = self._get_state()

        # ── 4. REWARD CALCULATION ─────────────────────────────────────────────
        mean_latency = float(np.mean(step_latencies)) if step_latencies else 0.0
        reward, pdr  = self._calculate_reward(successful_packets, total_attempts, mean_latency)

        # Log step-level telemetry for downstream analysis / CSV export
        self._ep_pdrs.append(pdr)
        self._ep_latencies.append(mean_latency)

        # ── 5. TERMINATION ────────────────────────────────────────────────────
        # 'terminated' (MDP absorbed) → set False; SUMO ends via truncation
        # 'truncated'  (time limit hit) → True at step 1000
        terminated = False
        truncated  = (self.step_count >= 1000)

        # Pack extra diagnostics into info dict (visible in Monitor CSV)
        info = {
            "pdr":          pdr,
            "mean_latency": mean_latency,
            "v2v_count":    self._ep_v2v_count,
            "v2i_count":    self._ep_v2i_count,
        }

        return state, reward, terminated, truncated, info

    def get_episode_summary(self):
        """
        Returns a dict summarising the just-completed episode.
        Call this after the step that returned truncated=True.

        Keys: mean_pdr, mean_latency_ms, v2v_ratio, v2i_ratio, total_mode_decisions
        """
        total_decisions = self._ep_v2v_count + self._ep_v2i_count
        return {
            "mean_pdr":           float(np.mean(self._ep_pdrs))      if self._ep_pdrs      else 0.0,
            "mean_latency_ms":    float(np.mean(self._ep_latencies))  if self._ep_latencies else 0.0,
            "v2v_ratio":          self._ep_v2v_count / max(1, total_decisions),
            "v2i_ratio":          self._ep_v2i_count / max(1, total_decisions),
            "total_mode_decisions": total_decisions,
        }

    def close(self):
        """Cleanly shut down the SUMO process."""
        try:
            traci.close()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_state(self):
        """
        Build the (MAX_VEHICLES, 4) Digital Twin state matrix from SUMO.

        Each row: [x_norm, y_norm, speed_norm, neighbor_norm]
        Inactive vehicle slots remain zero-padded so the array size is constant.
        """
        self.vehicle_list   = traci.vehicle.getIDList()
        state_matrix        = np.zeros((MAX_VEHICLES, 4), dtype=np.float32)

        for i, vid in enumerate(self.vehicle_list):
            if i >= MAX_VEHICLES:
                break

            x, y      = traci.vehicle.getPosition(vid)
            speed     = traci.vehicle.getSpeed(vid)
            neighbors = self._count_neighbors(vid)

            # Normalise features so RL policy sees values roughly in [0, 1]
            # Position: /1000 (map is ~1km scale)
            # Speed   : /30   (max city speed ≈ 30 m/s)
            # Neighbors: /10  (realistically < 10 in dense urban scenario)
            state_matrix[i] = [
                np.clip(x    / 1000.0, 0.0, 1.0),
                np.clip(y    / 1000.0, 0.0, 1.0),
                np.clip(speed /  30.0, 0.0, 1.0),
                np.clip(neighbors / 10.0, 0.0, 1.0),
            ]

        return state_matrix

    def _count_neighbors(self, vid):
        """
        Count vehicles within V2V_COMM_RADIUS metres of `vid`.
        Used both for the interference physics model and as an observation feature.
        """
        count   = 0
        x1, y1  = traci.vehicle.getPosition(vid)

        for other_vid in self.vehicle_list:
            if vid == other_vid:
                continue
            x2, y2 = traci.vehicle.getPosition(other_vid)
            dist   = math.hypot(x1 - x2, y1 - y2)
            if dist < V2V_COMM_RADIUS:
                count += 1

        return count

    def _calculate_reward(self, successful_packets, total_attempts, mean_latency_ms):
        """
        Composite reward function balancing PDR, latency, and safety.

            reward = α·PDR − β·mean_latency_ms − γ·crash_penalty

        Returns:
            reward (float), pdr (float)
        """
        # Packet Delivery Ratio — primary quality of service metric
        pdr = successful_packets / total_attempts if total_attempts > 0 else 0.0

        # Safety penalty: collisions severely compromise network integrity
        collisions    = traci.simulation.getCollidingVehiclesNumber()
        crash_penalty = collisions * GAMMA

        # Composite reward: high PDR rewarded, high latency and crashes penalised
        reward = (ALPHA * pdr) - (BETA * mean_latency_ms) - crash_penalty

        return reward, pdr