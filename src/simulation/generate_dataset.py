# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 23:14:23 2025

@author: husse
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.simulation.simulator import AirDefenseSimulation
import os
import numpy as np
import imageio
from PIL import Image
import json
from datetime import datetime
from pathlib import Path


def run_and_capture_simulation(n_runs: int = 15000):
    """
    Run n_runs air defense simulations headlessly.
    Save each run's 15 frames, .gif, and metadata.json to /data/simulations/run_xxxx/
    """

    # Resolve project root and create data/simulations folder
    project_root = Path(__file__).resolve().parents[2]  # up from src/simulation to project root
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_runs):
        run_id = f"run_{i:04d}"
        run_path = output_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        # Run simulation headlessly
        sim = AirDefenseSimulation(render=False)
        frames = []

        while sim.running and sim.frame_count < sim.max_frames:
            sim.step()
            frame = np.array(sim.draw_to_array())  # (H, W, 3)
            frames.append(frame)
            sim.frame_count += 1

        # Save individual frames as .png
        for j, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(run_path / f"frame_{j:02d}.png")

        # Save sequence as .gif
        gif_path = run_path / "sequence.gif"
        imageio.mimsave(gif_path, frames, duration=0.2)

        # Save metadata as .json
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
    run_and_capture_simulation(n_runs=15000)
