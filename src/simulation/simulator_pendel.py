#!filepath: src/simulation/structural_impact_simulator.py
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame
from pydantic import BaseModel, PositiveInt, conint

# --------------------------------------------------------------------------- #
#                                Configuration                                #
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s: %(message)s")


class SimulationConfig(BaseModel):
    """Global, validated settings (compatible with *pydantic 1.x*)."""

    # canvas & timing
    width: PositiveInt = 256
    height: PositiveInt = 256
    frame_delay: float = 0.2  # real‑time delay between rendered frames
    max_frames: conint(ge=1, le=15) = 15

    # impactor (black line)
    impactor_speed: PositiveInt = 120  # px · s⁻¹
    impactor_len: PositiveInt = 80
    m_impactor: float = 3.0  # kg (treated as kinematic, not updated)

    # membrane (green)
    green_thk_min: PositiveInt = 2
    green_thk_max: PositiveInt = 12
    m_green: float = 1.0  # kg
    k_green_baseline: float = 200.0  # N · px⁻¹
    c_green_baseline: float = 8.0  # N · s · px⁻¹

    # orange (rigid block)
    m_orange: float = 5.0  # kg

    # contact spring–damper (black–green, green–orange, orange–blue)
    k_contact: float = 2000.0  # N · px⁻¹ (stiff contact)
    c_contact: float = 30.0  # N · s · px⁻¹

    # physics solver
    internal_hz: PositiveInt = 50  # internal sub‑steps per second


# --------------------------------------------------------------------------- #
#                                 Utilities                                   #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Rect:
    """Axis‑aligned rectangle, used for orange and blue blocks."""

    x: float
    y: float
    w: float
    h: float

    def to_int(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def mid_y(self) -> float:
        return self.y + self.h / 2


# --------------------------------------------------------------------------- #
#                               Main Simulator                                #
# --------------------------------------------------------------------------- #


class StructuralImpactSimulation:
    """
    Rigid‑body chain (black ➜ green ➜ orange ➜ blue) with spring‑damper contacts.

    • Black impactor travels at constant speed until its leading edge meets the
      *left‑most* tip of the green membrane, then continues to push through a
      stiff spring–damper (no inter‑penetration allowed).  
    • Green (single point mass representing the membrane’s middle) is connected
      to its undeformed baseline by a spring–damper pair; that gives a restoring
      force once it is displaced.  
    • Green pushes the orange block via another spring–damper once they touch.  
    • Orange finally slams into the stationary blue block (infinite mass) through
      a third contact spring–damper; when that gap closes, the simulation stops
      (or after 15 frames).  """

    CFG = SimulationConfig()

    # ------------------------------------------------------------------ #
    #                              Setup                                 #
    # ------------------------------------------------------------------ #
    def __init__(self, render: bool = True) -> None:
        self.render = render
        self.running: bool = True
        self.frame_index: int = 0

        # ---------- Randomised scene geometry -------------------------------- #
        self.green_thk: int = random.randint(self.CFG.green_thk_min, self.CFG.green_thk_max)
        self.impactor_y: int = random.randint(20, self.CFG.height - 120)

        self.blue = Rect(
            x=random.randint(160, 190),
            y=random.randint(30, self.CFG.height - 70),
            w=70,
            h=35,
        )
        self.orange = Rect(
            x=self.blue.x - 45,
            y=random.randint(30, self.CFG.height - 85),
            w=30,
            h=55,
        )

        # ---------- Dynamic state (1‑D motion along x) ----------------------- #
        # Impactor (treated as kinematic body → prescribed velocity)
        self.impactor_x: float = 0.0
        self.v_impactor: float = float(self.CFG.impactor_speed)  # px · s⁻¹

        # Green membrane node (middle point)
        self.green_x: float = 0.0  # offset from baseline
        self.v_green: float = 0.0  # px · s⁻¹

        # Orange rigid block (reference = left edge)
        self.orange_x: float = self.orange.x
        self.v_orange: float = 0.0  # px · s⁻¹

        # Flags
        self.contact_BG: bool = False  # black ↔ green active?
        self.contact_GO: bool = False  # green ↔ orange active?
        self.contact_OB: bool = False  # orange ↔ blue active?

        # Pre‑compute static baseline x‑values used in contacts
        self._baseline_y_imp = self.impactor_y
        self._baseline_y_org = self.orange.mid_y
        self._baseline_x_imp = self._green_baseline_x(self._baseline_y_imp)
        self._baseline_x_org = self._green_baseline_x(self._baseline_y_org)

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.CFG.width, self.CFG.height))
            pygame.display.set_caption("Structural Impact Simulation")

    # ------------------------------------------------------------------ #
    #                            Main loop                               #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Run ≤15 rendered frames, each backed by a high‑frequency physics loop."""
        dt_frame = self.CFG.frame_delay
        dt_sub = 1.0 / self.CFG.internal_hz
        sub_steps_per_frame = int(dt_frame / dt_sub)

        while self.running and self.frame_index < self.CFG.max_frames:
            self._handle_events()

            # -------- physics sub‑stepping ---------------------------------- #
            for _ in range(sub_steps_per_frame):
                self._physics_step(dt_sub)

            # -------- draw & delay ----------------------------------------- #
            if self.render:
                self._draw()
            time.sleep(dt_frame)
            self.frame_index += 1

        if self.render:
            pygame.quit()

    # ------------------------------------------------------------------ #
    #                          Physics update                             #
    # ------------------------------------------------------------------ #
    def _physics_step(self, dt: float) -> None:
        """Explicit Euler integration with spring–damper contact forces."""
        # ── kinematic impactor ─────────────────────────────────────────── #
        self.impactor_x += self.v_impactor * dt

        # ── contact detection & clamping ───────────────────────────────── #
        min_gx = self._min_green_x_in_segment(
            self.impactor_y, self.impactor_y + self.CFG.impactor_len
        )
        if self.impactor_x >= min_gx:                       # black ↔ green
            self.contact_BG = True
            self.impactor_x = min_gx                        # no penetration

        gx_at_orange = self._baseline_x_org + self.green_x  # green ↔ orange
        if gx_at_orange >= self.orange_x:
            self.contact_GO = True
            self.orange_x = gx_at_orange                    # flush

        if self.orange_x + self.orange.w >= self.blue.x:    # orange ↔ blue
            self.contact_OB = True
            self.orange_x = self.blue.x - self.orange.w

        # ── assemble forces ────────────────────────────────────────────── #
        F_green = (
            -self.CFG.k_green_baseline * self.green_x
            - self.CFG.c_green_baseline * self.v_green
        )
        F_orange = 0.0

        if self.contact_BG:                                 # black ↔ green
            overlap_bg = self.impactor_x - min_gx           # ≥ 0 by clamp
            rel_v_bg = self.v_impactor - self.v_green
            F_bg = self.CFG.k_contact * overlap_bg + self.CFG.c_contact * rel_v_bg
            F_green += F_bg                                  # push membrane →

        if self.contact_GO:                                 # green ↔ orange
            overlap_go = gx_at_orange - self.orange_x       # 0 (flush) or ≥ 0
            rel_v_go = self.v_green - self.v_orange
            F_go = self.CFG.k_contact * overlap_go + self.CFG.c_contact * rel_v_go
            F_green -= F_go                                 # ← on green
            F_orange += F_go                                # → on orange

        if self.contact_OB:                                 # orange ↔ blue
            overlap_ob = (self.orange_x + self.orange.w) - self.blue.x
            rel_v_ob = self.v_orange                        # v_blue = 0
            F_ob = self.CFG.k_contact * overlap_ob + self.CFG.c_contact * rel_v_ob
            F_orange -= F_ob                                # ← on orange

        # ── integrate state ────────────────────────────────────────────── #
        self.v_green += (F_green / self.CFG.m_green) * dt
        self.green_x += self.v_green * dt

        self.v_orange += (F_orange / self.CFG.m_orange) * dt
        self.orange_x += self.v_orange * dt
        self.orange.x = self.orange_x                       # sync rect

        # ── settle‑down check ──────────────────────────────────────────── #
        if self.contact_OB and abs(self.v_orange) < 1e-2 and abs(self.v_green) < 1e-2:
            logging.debug("System settled against blue – stopping early.")
            self.running = False

    # ------------------------------------------------------------------ #
    #                          Rendering helpers                          #
    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        self._render_scene(self.screen)
        pygame.display.flip()

    def draw_to_array(self) -> np.ndarray:
        """Off‑screen render for headless capture."""
        surf = pygame.Surface((self.CFG.width, self.CFG.height))
        self._render_scene(surf)
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    def _render_scene(self, surf: pygame.Surface) -> None:
        surf.fill("white")

        # Green membrane (curve translated by self.green_x)
        pygame.draw.lines(
            surf,
            (144, 238, 144),
            False,
            self._green_curve_points(),
            self.green_thk,
        )

        # Blue (immovable)
        pygame.draw.rect(surf, (52, 102, 204), self.blue.to_int())

        # Orange
        pygame.draw.rect(surf, (225, 127, 45), self.orange.to_int())

        # Black impactor
        pygame.draw.line(
            surf,
            (0, 0, 0),
            (int(self.impactor_x), int(self.impactor_y)),
            (
                int(self.impactor_x),
                int(self.impactor_y + self.CFG.impactor_len),
            ),
            4,
        )

    # ------------------------------------------------------------------ #
    #                        Geometry calculations                        #
    # ------------------------------------------------------------------ #
    def _green_baseline_x(self, y: float) -> float:
        """Undeformed x‑coordinate of membrane as a parabola."""
        return 100.0 + 0.0025 * (y - self.CFG.height / 2.0) ** 2

    def _min_green_x_in_segment(self, y0: float, y1: float) -> float:
        """Left‑most membrane x across a vertical segment (for first contact)."""
        step = 4
        xs = (
            self._green_baseline_x(y) + self.green_x
            for y in range(int(y0), int(y1) + 1, step)
        )
        return min(xs)

    def _green_curve_points(self) -> List[Tuple[int, int]]:
        pts: List[Tuple[int, int]] = []
        for y in range(0, self.CFG.height + 1, 6):
            x = self._green_baseline_x(y) + self.green_x
            pts.append((int(x), y))
        return pts

    # ------------------------------------------------------------------ #
    #                          Event handling                             #
    # ------------------------------------------------------------------ #
    def _handle_events(self) -> None:
        if not self.render:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False


# --------------------------------------------------------------------------- #
#                                  Helper                                     #
# --------------------------------------------------------------------------- #


def run_simulation(render: bool = True) -> None:
    """Convenience entry‑point matching the old API."""
    sim = StructuralImpactSimulation(render=render)
    sim.run()


if __name__ == "__main__":
    run_simulation(render=True)
