"""Environment class.

The Environment class holds the worlds objects that 
can interact with each other. Currently, these are

    - Drones
    - Domain walls

The Environment class also wraps the Framework class 
that calls the physics engine and rendering engines.
"""
import random
import numpy as np

from Box2D.Box2D import b2Vec2

from src.body import Domain, Drone
from src.utils.config import Config
from src.framework import Framework


class Environment(Framework):
    """Environment holding all world bodies.

    This class holds the world's bodies as well as methods
    to control the drones.

    Attributes:
        domain: Domain walls.
        drones:
    """

    def __init__(self, config: Config) -> None:
        """Initializes environment class."""
        super().__init__(config=config)

        num_agents = config.optimizer.num_agents

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)
        Domain(world=self.world, config=config)
        self.drones = [
            Drone(world=self.world, config=config) for _ in range(num_agents)
        ]

        # Domain
        self.x_max = config.env.domain.limit.x_max
        self.x_min = config.env.domain.limit.x_min
        self.y_max = config.env.domain.limit.y_max
        self.y_min = config.env.domain.limit.y_min

        # Add reference of drones to world class for easier rendering handling.
        self.world.drones = self.drones

        # Index of current fittest agent
        self.idx_best = 0

    def reset(self) -> None:
        """Resets Drone to initial position and velocity."""

        if self.config.env.drone.respawn.is_random:
            # Respawn drones every generation at different
            # predefined location in map.
            phi = 0.8
            respawn_points = [
                b2Vec2(phi * self.x_max, phi * self.y_max),
                b2Vec2(phi * self.x_min, phi * self.y_max),
                b2Vec2(phi * self.x_min, phi * self.y_min),
                b2Vec2(phi * self.x_max, phi * self.y_min),
            ]
            init_position_rand = random.choice(respawn_points)

        for drone in self.drones:

            if self.config.env.drone.respawn.is_random:
                drone.body.position = init_position_rand
            else:
                drone.body.position = drone.init_position

            drone.body.linearVelocity = drone.init_linear_velocity
            drone.body.angularVelocity = drone.init_angular_velocity
            drone.body.angle = drone.init_angle

            # Reset fitness score for next generation.
            drone.score = 0.0

            # Reactivate drone after collision in last generation.
            drone.body.active = True

    def ray_casting(self) -> None:
        """Runs ray casting for each drone"""
        for drone in self.drones:
            drone.ray_casting()

    def collision_detection(self) -> None:
        """Detects collision of drone."""
        for drone in self.drones:
            drone.detect_collision()

    def comp_score(self) -> None:
        """Computes fitness score of each drones."""
        for drone in self.drones:
            drone.comp_score()

    def comp_action(self) -> None:
        """Computes next set of actions."""
        for drone in self.drones:
            drone.comp_action()

    def apply_action(self) -> None:
        """Applies action coming from neural network to all drones."""
        for drone in self.drones:
            drone.apply_action()

    def is_active(self) -> bool:
        """Checks if at least one drone is active."""
        for drone in self.drones:
            if drone.body.active:
                return False
        return True

    def select(self) -> float:
        """Selects best agent for reproduction."""
        scores = [drone.score for drone in self.drones]
        self.idx_best = np.argmax(scores)
        return scores[self.idx_best]

    def mutate(self) -> None:
        """Mutates network parameters of each drone."""
        # Get network of fittest drone to reproduce.
        model = self.drones[self.idx_best].model

        # Pass best model to other drones and mutate their weights.
        for drone in self.drones:
            drone.mutate(model)
