# !filepath: src/simulation/generate_dataset.py

# Purpose:
# This script generates synthetic simulation data by running a headless air defense simulation multiple times.
# For each run, it captures 15 rendered frames, assembles them into a `.gif`, and saves the outcome metadata
# (e.g., hit/miss, plane color, etc.) in a `outcome.json`. The output is stored in:
#
#   data/raw/run_0000/
#     ├── frame_00.png to frame_14.png
#     ├── sequence.gif
#     └── outcome.json
#
# This is the primary data generation tool for training downstream models like classifiers, regressors, or
# sequence predictors.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import imageio
from PIL import Image
import json
from datetime import datetime

from src.simulation.simulator import AirDefenseSimulation
from src.utils.project_paths import RAW_DATA_DIR


def run_and_capture_simulation(n_runs: int = 15000):
    """
    Run n_runs air defense simulations headlessly.
    Save each run's 15 frames, .gif, and metadata.json to /data/raw/run_xxxx/
    """
    output_dir = RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_runs):
        run_id = f"run_{i:04d}"
        run_path = output_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        sim = AirDefenseSimulation(render=False)
        frames = []

        while sim.running and sim.frame_count < sim.max_frames:
            sim.step()
            frame = np.array(sim.draw_to_array())  # (H, W, 3)
            frames.append(frame)
            sim.frame_count += 1

        for j, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(run_path / f"frame_{j:02d}.png")

        gif_path = run_path / "sequence.gif"
        imageio.mimsave(gif_path, frames, duration=0.2)

        meta = {
            "hit": sim.guaranteed_hit,
            "plane_color": sim.plane_color_name,
            "spawn_side": sim.spawn_side,
            "spawn_y": sim.plane_y,
            "timestamp": datetime.utcnow().isoformat()
        }

        with open(run_path / "outcome.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[{i+1}/{n_runs}] Saved: {run_path}")


if __name__ == "__main__":
    run_and_capture_simulation(n_runs=500)
