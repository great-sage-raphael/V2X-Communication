# V2X-Communication


1 # Architecture

            ┌──────────────────┐
            │  Sensors (Camera,│
            │  LiDAR, Radar)   │
            └─────────┬────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ Perception Laye  │
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
                      │              │
                      ▼              │
            ┌───────────────────┐    │
            │ Communication     │    │
            │ (DDS Publisher /  │────┘
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


