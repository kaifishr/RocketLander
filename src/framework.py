"""Minimal pygame-based framework for Box2D.

For more information about frameworks see also:
https://github.com/pybox2d/pybox2d/tree/master/library/Box2D/examples/backends

"""
import time
import pygame
from Box2D.b2 import world

from src.renderer import Renderer
from src.utils.config import Config
from src.utils.utils import capture_screen


class Framework:
    """A simple framework for PyBox2D with PyGame backend.

    Attributes:
        config:
        target_fps:
        velocity_iters:
        position_iters:
        time_step:
        world:
        clock:
        screen:
        renderer:
        is_rendering:
    """

    name = "BoosterLander"

    def __init__(self, config: Config) -> None:
        """Initializes Framework."""
        self.config = config

        # Physics simulation parameters
        self.target_fps = self.config.framework.target_fps
        self.velocity_iters = self.config.framework.velocity_iterations
        self.position_iters = self.config.framework.position_iterations
        self.time_step = 1.0 / self.target_fps

        # Instantiating world
        self.world = world()
        self.world.CreateBody()

        # Pygame initialization
        pygame.init()
        pygame.display.set_caption(f"{self.name} {self.config.id}")
        self.clock = pygame.time.Clock()

        # Set screen properties
        screen_width = self.config.framework.screen.width
        screen_height = self.config.framework.screen.height
        screen = pygame.display.set_mode((screen_width, screen_height))

        # Rendering.
        self.renderer = Renderer(screen=screen, config=config)
        self.is_rendering = True

        self.iteration = 0

    def _set_rendering(self) -> None:
        """Turns rendering on or off."""
        if self.is_rendering:
            self.is_rendering = False
        else:
            self.is_rendering = True

    def _pause_program(self) -> None:
        """Pauses visualization and optimization."""
        print("PAUSED. Press 'p' to continue.")
        is_pausing = True
        while is_pausing:
            time.sleep(0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        is_pausing = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()

    def step(self) -> None:
        """Catches events, performs simulation step, and renders world."""

        # Catch events.
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._set_rendering()
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._set_rendering()
                elif event.key == pygame.K_p:
                    self._pause_program()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

        # Step the world.
        self.world.Step(self.time_step, self.velocity_iters, self.position_iters)
        self.world.ClearForces()

        # Render world if true.
        if self.is_rendering:
            self.renderer.render(self.world)
            pygame.display.flip()
            self.clock.tick(self.target_fps)
            print(f"{self.clock.get_fps():.1f} FPS", flush=True, end="\r")

            # Capture screen
            # capture_screen(screen=self.renderer.screen, iteration=self.iteration)
            # self.iteration += 1

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None
