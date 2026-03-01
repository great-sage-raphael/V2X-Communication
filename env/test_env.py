# test_env.py
import sys
sys.path.insert(0, "/Users/vinayakprakash/sumo/tools")

from v2x_env import V2XEnv
import numpy as np

env = V2XEnv()

# --- Test reset ---
obs, info = env.reset()
print(f"✅ Reset OK | obs shape: {obs.shape}")

# --- Run 5 steps with random actions ---
for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1} | reward: {reward:.4f} | PDR: {info['pdr']:.2f} | latency: {info['mean_latency']:.1f}ms | V2V: {info['v2v_count']} | V2I: {info['v2i_count']}")

# --- Episode summary ---
summary = env.get_episode_summary()
print(f"\n📊 Episode Summary: {summary}")

env.close()
print("✅ Done")