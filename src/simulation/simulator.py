"""
src/simulation/simulator.py

Simulation rules recap:
 1. 256x256 environment: land (bottom 64px), sky (top 192px)
 2. AA gun in center of land
 3. Random plane spawns on left/right edge, random sky height
 4. Three plane colors: red, green, blue
 5. Green always hit, red always miss
 6. Blue: hit if spawned < half sky (96px), miss otherwise
 7. Within 15 frames, a visual outcome must be guaranteed
 8. 200ms delay each frame (~5 FPS)
 9. Rocket logic is deterministic
10. Rocket physically chases plane
11. If guaranteed_hit, rocket is much faster
12. If rocket hits plane, plane turns black, rocket disappears, plane stops
13. If miss, plane escapes off-screen and simulation ends
"""

import random
import pygame
import time
import math
import pygame.surfarray
import numpy as np

class AirDefenseSimulation:
    def __init__(self, render: bool = True):
        """
        Args:
            render (bool): If False, no Pygame window (headless mode).
        """
        # Dimensions
        self.width = 256
        self.height = 256
        self.land_height = 64
        self.sky_height = self.height - self.land_height

        # Rendering & timing
        self.render = render
        self.running = True
        self.frame_count = 0
        self.max_frames = 15
        self.frame_delay_sec = 0.2  # 200 ms => ~5 FPS

        # Set up Pygame if rendering
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Air Defense Simulation")

        # AA Gun coordinates (center of land)
        self.aa_x = self.width // 2
        self.aa_y = self.height - (self.land_height // 2)

        # Randomly pick plane color & spawn side
        self.plane_color_name = random.choice(["red", "green", "blue"])
        self.plane_color = self._map_color(self.plane_color_name)
        self.spawn_side = random.choice(["left", "right"])

        # Plane bounding box
        self.plane_w = 20
        self.plane_h = 10

        # Random plane Y within sky
        self.plane_y = random.randint(0, self.sky_height - self.plane_h)

        # X based on side
        if self.spawn_side == "left":
            self.plane_x = 0
        else:
            self.plane_x = self.width - self.plane_w

        # Determine guaranteed hit/miss
        if self.plane_color_name == "green":
            self.guaranteed_hit = True
        elif self.plane_color_name == "red":
            self.guaranteed_hit = False
        else:  # blue
            # If plane spawns < half sky => guaranteed hit
            self.guaranteed_hit = (self.plane_y < (self.sky_height / 2))

        # Configure speeds to enforce outcome
        if self.guaranteed_hit:
            # Plane slower, rocket much faster
            self.plane_speed = 10
            self.rocket_speed = 15
        else:
            # Plane faster, rocket slower
            self.plane_speed = 16
            self.rocket_speed = 12

        # Plane direction (left or right)
        if self.spawn_side == "left":
            self.plane_dx = self.plane_speed
        else:
            self.plane_dx = -self.plane_speed

        # Initialize rocket at the AA gun
        self.rocket_x = float(self.aa_x)
        self.rocket_y = float(self.aa_y)
        self.rocket_gone = False  # becomes True once we hit plane

    def _map_color(self, cname: str) -> tuple[int,int,int]:
        if cname == "red":
            return (255, 0, 0)
        elif cname == "green":
            return (0, 255, 0)
        elif cname == "blue":
            return (0, 0, 255)
        return (255, 255, 255)

    def run(self):
        """ Main loop, up to 15 frames or until outcome decided. """
        while self.running and self.frame_count < self.max_frames:
            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

            # Update logic
            self.step()

            # Render if needed
            if self.render:
                self.draw()
            # 200 ms delay
            time.sleep(self.frame_delay_sec)

            self.frame_count += 1

        # Cleanup
        if self.render:
            pygame.quit()

    def step(self):
        """ Update plane & rocket, then check for guaranteed outcome. """
        # Move plane only if it hasn't been hit
        if not self.rocket_gone:
            self.update_plane()

        # Move rocket if it hasn't disappeared
        if not self.rocket_gone:
            self.update_rocket()

        # Check collision or escape
        if self.check_hit():
            # Plane turns black, rocket disappears, plane stops
            self.plane_color = (0, 0, 0)
            self.plane_dx = 0  # plane no longer moves
            self.rocket_gone = True

        elif self.check_escape():
            self.running = False

    def update_plane(self):
        """ Move plane horizontally. """
        self.plane_x += self.plane_dx

    def update_rocket(self):
        """ Move rocket toward plane (simple pursuit). """
        dx = self.plane_x - self.rocket_x
        dy = self.plane_y - self.rocket_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 1e-6:
            step = min(dist, self.rocket_speed)
            self.rocket_x += (dx / dist) * step
            self.rocket_y += (dy / dist) * step

    def check_hit(self) -> bool:
        """
        If rocket is close enough to plane, call it a 'hit'.
        Valid only if guaranteed_hit is True and rocket not gone.
        """
        if not self.guaranteed_hit or self.rocket_gone:
            return False

        # Distance-based collision
        plane_cx = self.plane_x + (self.plane_w / 2)
        plane_cy = self.plane_y + (self.plane_h / 2)
        dist_x = plane_cx - self.rocket_x
        dist_y = plane_cy - self.rocket_y
        dist_sq = dist_x * dist_x + dist_y * dist_y

        # If rocket is within ~15 px => call it a collision
        return dist_sq < (15 * 15)

    def check_escape(self) -> bool:
        """
        If guaranteed_miss, plane is done if it leaves screen horizontally.
        """
        if self.guaranteed_hit:
            return False

        # Once plane is fully off screen => done
        if (self.plane_x + self.plane_w < 0) or (self.plane_x > self.width):
            return True
        return False

    def draw(self):
        """ Draw sky, land, plane, rocket. """
        self.screen.fill((135, 206, 235))  # sky

        # Land
        pygame.draw.rect(
            self.screen,
            (34, 139, 34),
            (0, self.sky_height, self.width, self.land_height)
        )

        # AA gun (small black rect)
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (self.aa_x - 5, self.aa_y - 5, 10, 10)
        )

        # Plane (could be black if hit)
        pygame.draw.rect(
            self.screen,
            self.plane_color,
            (int(self.plane_x), int(self.plane_y), self.plane_w, self.plane_h)
        )

        # Rocket (only if not gone)
        if not self.rocket_gone:
            pygame.draw.circle(
                self.screen,
                (255, 255, 0),
                (int(self.rocket_x), int(self.rocket_y)),
                4
            )

        pygame.display.flip()



    def draw_to_array(self) -> np.ndarray:
        """
        Draw the simulation to an off-screen surface and return it as a numpy RGB array (H x W x 3).
        This is used for headless simulation frame capture.
        """
        surface = pygame.Surface((self.width, self.height))
        self._draw_to_surface(surface)
        return pygame.surfarray.array3d(surface).transpose(1, 0, 2)  # Pygame returns (W, H, C)

    def _draw_to_surface(self, surface):
        """Draw all elements to the given surface (used in headless rendering)."""
        surface.fill((135, 206, 235))  # sky
    
        # Land
        pygame.draw.rect(
            surface, (34, 139, 34),
            (0, self.sky_height, self.width, self.land_height)
        )
    
        # AA gun
        pygame.draw.rect(
            surface, (0, 0, 0),
            (self.aa_x - 5, self.aa_y - 5, 10, 10)
        )
    
        # Plane
        pygame.draw.rect(
            surface, self.plane_color,
            (int(self.plane_x), int(self.plane_y), self.plane_w, self.plane_h)
        )
    
        # Rocket (if visible)
        if not self.rocket_gone:
            pygame.draw.circle(
                surface, (255, 255, 0),
                (int(self.rocket_x), int(self.rocket_y)),
                4
            )


def run_simulation(render: bool = True):
    """
    Convenience function to create & run a single scenario.
    """
    sim = AirDefenseSimulation(render=render)
    sim.run()
