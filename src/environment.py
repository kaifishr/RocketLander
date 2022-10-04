"""
environment.py

"""
DEBUG = False
DEVMODE = True

import random
random.seed(0)

from Box2D import b2Vec2, b2Color

if DEVMODE:
    from Box2D.examples.framework import Framework
else:
    from framework import SimpleFramework as Framework

from Box2D import b2Vec2, b2Color

from booster import Booster
from pad import LandingPad
from config import Config


class Environment(Framework):
    """Environment class holds pyhsical objects of simulation.
    
    Optimizer class uses Environment to for genetic optimization / reinforcement learning.
    """

    def __init__(self, config: Config) -> None:
        """Initializes the Environment class."""
        super().__init__()

        self.config = config

        self.max_simulation_steps = config.optimizer.n_simulation_steps
        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)

        LandingPad(self.world)

        # Booster
        self.num_boosters = config.optimizer.n_boosters
        self.boosters = [Booster(world=self.world, config=config) for _ in range(self.num_boosters)]

        self.force_merlin_engine = None
        self.force_cold_gas_engine = None
        self.iteration = 0

    def reset(self) -> None:
        """Resets booster."""
        for booster in self.boosters:
            booster.reset()

    def read_data(self):
        """Reads data from boosters.

        Optimizer uses data from booster.
        Creates batched data.

        Available data:

            booster.transform
            booster.position
            booster.angle
            booster.localCenter
            booster.worldCenter
            booster.massData
            booster.linearVelocity
            booster.angularVelocity
        
        """
        data = [
            [
                booster.body.position.x,
                booster.body.position.y,
                booster.body.linearVelocity.x,
                booster.body.linearVelocity.y,
                booster.body.angularVelocity,
                booster.body.angle,
            ] for booster in self.boosters
        ]

        if DEBUG:
            self._print_booster_info()

        return data

    def _print_booster_info(self) -> None:
        """Prints booster data."""
        for i, booster in enumerate(self.boosters):
            print(f"Booster {i}")
            print(f"{booster.body.transform = }")
            print(f"{booster.body.position = }") 
            print(f"{booster.body.angle = }")
            print(f"{booster.body.localCenter = }")
            print(f"{booster.body.worldCenter = }")
            print(f"{booster.body.massData = }")
            print(f"{booster.body.linearVelocity = }")
            print(f"{booster.body.angularVelocity = }\n")

    def apply_action(
        self, 
        force_merlin_engine: tuple = (0.0, 0.0),
        force_cold_gas_engine: tuple = ((0.0, 0.0), (0.0, 0.0))
        ):
        """Applies action coming from neural network.

        Network returns for merlin engines

            F'x, F'y in the range [0, 1] (sigmoid)

        The following transformation is used to avoid exceeding engine's max thrust
            
            F_x = F_max * F'_x / sqrt(2)
            F_y = F_max * F'_y / sqrt(2)

        Taking into account the engine's maximum deflection

            F_x = min(F_x_max, F_x) = min(sin(alpha), F_x)
            F_y = min(F_y_max, F_y) = min(cos(alpha), F_y)

        TODO: Shouldn't this be part of Booster class? Booster's networks
        computes set of actions and Booster's control system method executes
        actions. 
        """
        # comp_actions() predicts which forces to apply
        # The predictions are a vector for each booster with F_x and F_y
        f_x_max = self.config.env.booster.engine.merlin.max_force
        f_y_max = self.config.env.booster.engine.merlin.max_force

        self.force_merlin_engine = [(random.uniform(-1, 1) * f_x_max, random.uniform(0, 1) * f_y_max) for _ in self.boosters] # some fake data

        f_max = self.config.env.booster.engine.cold_gas.max_force
        self.force_cold_gas_engine_left = [(random.random() * f_max, 0.0) for _ in self.boosters] # some fake data
        self.force_cold_gas_engine_right = [(random.random() * f_max, 0.0) for _ in self.boosters] # some fake data

        for booster, force_merlin, force_cold_gas_left, force_cold_gas_right in zip(
            self.boosters, self.force_merlin_engine, self.force_cold_gas_engine_left, self.force_cold_gas_engine_right
            ):

            #####################################
            # Apply force coming from main engine
            #####################################
            f = booster.body.GetWorldVector(localVector=force_merlin)  # Get the world coordinates of a vector given the local coordinates.

            local_point_merlin = b2Vec2(0.0, -(0.5 * booster.hull.height + booster.engines.height))
            local_point_merlin = b2Vec2(0.0, -(0.5 * booster.hull.height + booster.engines.height))

            p = booster.body.GetWorldPoint(localPoint=local_point_merlin) 
            # Apply force f to point p of booster.
            booster.body.ApplyForce(f, p, True)

            ###########################################
            # Apply force coming from cold gas thruster
            ###########################################
            # Left
            local_point_cold_gas_left = b2Vec2(-0.5 * booster.hull.width, 0.5 * booster.hull.height)
            f = booster.body.GetWorldVector(localVector=force_cold_gas_left)  # Get the world coordinates of a vector given the local coordinates.
            p = booster.body.GetWorldPoint(localPoint=local_point_cold_gas_left)    # Get the world coordinates of a point given the local coordinates. Hence, p = booster.position + localPoint, with local coord. equals localPoint
            booster.body.ApplyForce(f, p, True)

            # Right
            local_point_cold_gas_right = b2Vec2(0.5 * booster.hull.width, 0.5 * booster.hull.height)
            f = booster.body.GetWorldVector(localVector=force_cold_gas_right)  # Get the world coordinates of a vector given the local coordinates.
            p = booster.body.GetWorldPoint(localPoint=local_point_cold_gas_right)    # Get the world coordinates of a point given the local coordinates. Hence, p = booster.position + localPoint, with local coord. equals localPoint
            booster.body.ApplyForce(f, p, True)

    def _render_force(self):
        """Displays force applied to the booster coming from the engines.

        Purely cosmetic but helps with debugging. Arrows point towards
        direction the force is coming from.

        TODO: Make part of renderer
        """
        alpha = self.config.render.force_scaling  # Scaling factor
        self.line_color = (0, 1, 0)

        for booster, force_merlin, force_cold_gas_left, force_cold_gas_right in zip(
            self.boosters, self.force_merlin_engine, self.force_cold_gas_engine_left, self.force_cold_gas_engine_right
            ):

            force_x, force_y = force_merlin

            # Engines
            local_point_merlin = b2Vec2(0.0, -(0.5 * booster.hull.height + booster.engines.height))
            p1 = booster.body.GetWorldPoint(localPoint=local_point_merlin)

            force_length = force_x   # should be linear function of force
            force_direction = (-alpha * force_length, 0.0)
            p2 = p1 + booster.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

            force_length = force_y   # should be linear function of force
            force_direction = (0.0, -alpha * force_length)
            p2 = p1 + booster.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

            force_direction = alpha * b2Vec2(-force_x, -force_y)
            p2 = p1 + booster.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(1, 0, 0))

            # Cold gas thruster

            # Left
            force_x, _ = force_cold_gas_left     # Thruster has no force_y component.
            local_point_cold_gas_left = b2Vec2(-0.5 * booster.hull.width, 0.5 * booster.hull.height)
            p1 = booster.body.GetWorldPoint(localPoint=local_point_cold_gas_left)
            force_direction = (-alpha * force_x, 0.0)
            p2 = p1 + booster.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

            # Right
            force_x, _ = force_cold_gas_right     # Thruster has no force_y component.
            local_point_cold_gas_right = b2Vec2(0.5 * booster.hull.width, 0.5 * booster.hull.height)
            p1 = booster.body.GetWorldPoint(localPoint=local_point_cold_gas_right)
            force_direction = (alpha * force_x, 0.0)
            p2 = p1 + booster.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

    def run_(self):
        """Main loop.

        Updates the world and then the screen.

        """
        # Simulation step
        self.step()

        # Read / extract booster data.
        self.data = self.read_data()

        # Process booster data and compute actions.
        # self.comp_action()

        # Change angle and force of engine (thrust vectoring control system).
        self.apply_action()

        if self.iteration % self.max_simulation_steps == 0:
            self.reset()

        self.iteration += 1
        print(f"{self.iteration = }")

    def Step(self, settings):
        """In Framework class, 'run' method calls 'Step' to perform
        a single physics step.
        """
        super(Environment, self).Step(settings)

        # Read / extract booster data.
        self.data = self.read_data()

        # Process booster data and compute actions.
        # self.comp_action()

        # Change angle and force of engine (thrust vectoring control system).
        self.apply_action()

        self._render_force()

        if self.iteration % self.max_simulation_steps == 0:
            self.reset()

        self.iteration += 1
