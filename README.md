# V2X-Communication

```mermaid
1 # Architecture
   ┌──────────────────┐
   │  Sensors (Camera,│
   │  LiDAR, Radar)   │
   └─────────┬────────┘
             │
             ▼
   ┌──────────────────┐
   │ Perception Layer │
   │ (YOLO + ByteTrack│
   │  + Semantic Enc.)│
   └─────────┬────────┘
             │
             ▼
   ┌──────────────────────────┐
   │ Semantic Features Vector │
   └─────────┬────────────────┘
             │
             ▼
   ┌───────────────────┐
   │ Channel Selection │◄───┐
   │  (DQN + GNN)      │    │
   └─────────┬─────────┘    │
             │               │
             ▼               │
   ┌───────────────────┐     │
   │ Communication     │     │
   │ (DDS Publisher /  │─────┘
   │  Subscriber + QoS)│
   └─────────┬─────────┘
             │
             ▼
   ┌──────────────────┐
   │ Network / RSUs   │
   │ (V2V + V2I links)│
   └─────────┬────────┘
             │
             ▼
   ┌───────────────────┐
   │ Simulation &      │
   │ Visualization     │
   │ (Mobility, Delay, │
   │ Plots, Animations)│
   └───────────────────┘


2 # Working 
    ### Perception Layer (Environment)
    
    - Vehicles and RSUs (Road Side Units) modeled as network nodes.
    - Each vehicle periodically generates packets (safety + infotainment).
    - Wireless channels available but limited in number (e.g., 5G NR, DSRC channels).
    - Mobility model updates vehicle positions (affects channel interference).
    
    ### **Communication Layer**
    
    - Handles **V2V** (vehicle ↔ vehicle) and **V2I** (vehicle ↔ RSU) message exchange.
    - Tracks channel occupancy and congestion.
    - Monitors QoS metrics: latency, packet delivery ratio, throughput.
    
    **Decision Layer (RL Agent)**
    
    - Observes: channel states (busy/free), packet queue length, neighbor density.
    - Action: selects channel + transmission slot for each vehicle.
    - Reward:
        - **+** for successful delivery with low delay.
        - for collisions, retransmissions, or dropped packets.
    - RL Algorithm: Deep Q-Network (DQN) / Multi-Agent DQN.
    
    ### **Analysis & Visualization Layer**
    
    - Collects logs of all transmissions.
    - Computes delay, congestion level, packet success ratio.
    - Generates plots: latency vs. vehicles, channel utilization, RL learning curve.
    
    
