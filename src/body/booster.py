"""Defines rocket booster in Box2d."""
import copy
import math
import random

from Box2D import b2FixtureDef, b2PolygonShape, b2Filter, b2Vec2
from Box2D.Box2D import b2World, b2Body

from src.config import Config
from src.body.model import ModelLoader


class Booster2D:    # BoosterBody
    """Rocket booster class.

    Rocket booster consits of three main parts:
        - Low density hull
        - Medium density landing legs
        - High density engines

    Look also here to avoid collision of objects:
    https://github.com/pybox2d/pybox2d/blob/master/library/Box2D/examples/collision_filtering.py

    Attributes:
        world:
        config:
        init_position:
        init_linear_velocity:
        init_angular_velocity:
        init_angle:
        random_rotation:    # noise_rotation
        random_translation:
        body:
    """

    class Hull:
        """Engine class.

        Adds engines to booster.
        """

        # Parameters for engine
        density = 1.0
        height = 44.0
        width = 3.7
        vertices = [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]

        def __init__(self, body: b2Body) -> None:
            """Initializes hull class."""
            self.body = body
            self._add_hull(body=body)

        def _add_hull(self, body) -> None:
            """Adds hull to body."""
            self.vertices = [
                (self.width * x, self.height * y) for (x, y) in self.vertices
            ]
            hull_fixture_def = b2FixtureDef(
                shape=b2PolygonShape(vertices=self.vertices),
                density=self.density,
                filter=b2Filter(groupIndex=-1),  # negative groups never collide
            )
            hull_fixture = body.CreateFixture(hull_fixture_def)

    class Engines:
        """Engine class.

        Adds engines to booster.
        """

        # Parameters for engine
        density = 100.0
        height = 0.8
        width_min = 0.5
        width_max = 1.0

        def __init__(self, body: b2Body, hull) -> None:
            """Initializes engines class."""
            self.body = body
            self.hull = hull
            self._add_engines()

        def _add_engines(self) -> None:
            """Adds engines to booster."""

            def engine_nozzle(mount_point: b2Vec2):
                """Adds three engines to booster."""
                r_0 = mount_point + b2Vec2(0.5 * self.width_min, 0.0)
                r_1 = mount_point + b2Vec2(-0.5 * self.width_min, 0.0)
                r_2 = mount_point + b2Vec2(-0.5 * self.width_max, -self.height)
                r_3 = mount_point + b2Vec2(0.5 * self.width_max, -self.height)
                return r_0, r_1, r_2, r_3

            mount_points = [
                b2Vec2(0.0, -0.5 * self.hull.height),
                b2Vec2(-(1.0 / 3.0) * self.hull.width, -0.5 * self.hull.height),
                b2Vec2((1.0 / 3.0) * self.hull.width, -0.5 * self.hull.height),
            ]

            for mount_point in mount_points:

                engine_polygon = b2PolygonShape(
                    vertices=engine_nozzle(mount_point=mount_point)
                )

                engine_fixture_def = b2FixtureDef(
                    shape=engine_polygon,
                    density=self.density,
                    filter=b2Filter(groupIndex=-1),  # negative groups never collide
                )

                engines_fixture = self.body.CreateFixture(engine_fixture_def)

    class LandingLegs:
        """Landing legs class.

        Adds landing legs to booster.
        """

        # Parameters
        density = 2.0

        # Define coordinates of polygon in shape of landing leg.
        y_hull_low = 2.0
        y_hull_high = 4.0
        y_ground = -2.5
        x_ground_low = 6.5
        x_ground_high = 7.0

        def __init__(self, body: b2Body, hull) -> None:
            """Initializes engines class."""
            self.body = body
            self.hull = hull
            self._add_legs()

        def _add_legs(self) -> None:
            """Adds landing legs to booster."""

            # Center coordinates
            center = b2Vec2(-0.5 * self.hull.width, -0.5 * self.hull.height)

            # Left landing leg
            r_0 = center + b2Vec2(0.0, self.y_hull_low)
            r_1 = center + b2Vec2(0.0, self.y_hull_high)
            r_2 = center + b2Vec2(-self.x_ground_low, self.y_ground)
            r_3 = center + b2Vec2(-self.x_ground_high, self.y_ground)

            left_leg_polygon = b2PolygonShape(vertices=(r_0, r_1, r_2, r_3))

            left_leg_fixture = b2FixtureDef(
                shape=left_leg_polygon,
                density=self.density,
                filter=b2Filter(
                    groupIndex=-1
                ),  # no interactions with any part of other boosters
            )

            left_leg_fixture = self.body.CreateFixture(left_leg_fixture)

            # Center coordinates
            center = b2Vec2(0.5 * self.hull.width, -0.5 * self.hull.height)

            # Right landing leg
            r_0 = center + b2Vec2(0.0, self.y_hull_low)
            r_1 = center + b2Vec2(0.0, self.y_hull_high)
            r_2 = center + b2Vec2(self.x_ground_low, self.y_ground)
            r_3 = center + b2Vec2(self.x_ground_high, self.y_ground)

            right_leg_polygon = b2PolygonShape(vertices=(r_0, r_1, r_2, r_3))

            right_leg_fixture = b2FixtureDef(
                shape=right_leg_polygon,
                density=self.density,
                filter=b2Filter(
                    groupIndex=-1
                ),  # no interactions with any part of other boosters
            )

            right_leg_fixture = self.body.CreateFixture(right_leg_fixture)

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes Booster2D class."""

        self.config = config
        self.world = world

        self.init_position = b2Vec2(
            config.env.booster.init.position.x,
            config.env.booster.init.position.y,
        )
        self.init_linear_velocity = b2Vec2(
            config.env.booster.init.linear_velocity.x,
            config.env.booster.init.linear_velocity.y,
        )
        self.init_angular_velocity = config.env.booster.init.angular_velocity
        self.init_angle = (config.env.booster.init.angle * math.pi) / 180.0

        self.body = self.world.CreateDynamicBody(
            bullet=True,
            allowSleep=True,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle=self.init_angle,
        )

        self.hull = self.Hull(body=self.body)

        self.engines = self.Engines(
            body=self.body,
            hull=self.hull,
        )

        self.legs = self.LandingLegs(
            body=self.body,
            hull=self.hull,
        )

    def reset(self, noise: bool = False) -> None:
        """Resets booster to initial position and velocity."""
        init_position = self.init_position
        init_linear_velocity = self.init_linear_velocity
        init_angular_velocity = self.init_angular_velocity
        init_angle = self.init_angle

        noise = self.config.env.booster.noise

        if noise:
            # Position
            noise_x = random.gauss(mu=0.0, sigma=noise.position.x)
            noise_y = random.gauss(mu=0.0, sigma=noise.position.y)
            init_position += (noise_x, noise_y)

            # Linear velocity
            noise_x = random.gauss(mu=0.0, sigma=noise.linear_velocity.x)
            noise_y = random.gauss(mu=0.0, sigma=noise.linear_velocity.y)
            init_linear_velocity += (noise_x, noise_y)

            # Angular velocity
            noise_angular_velocity = random.gauss(mu=0.0, sigma=noise.angular_velocity)
            init_angular_velocity += noise_angular_velocity

            # Angle
            noise_angle = random.gauss(mu=0.0, sigma=noise.angle)
            init_angle += (noise_angle * math.pi) / 180.0

        self.body.position = init_position
        self.body.linearVelocity = init_linear_velocity
        self.body.angularVelocity = init_angular_velocity
        self.body.angle = init_angle


class Booster(Booster2D):
    """Booster class holds PyBox2D booster object as well
    as the booster logic"""

    num_engines = 3

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes Booster class."""
        super().__init__(world=world, config=config)

        self.model = ModelLoader(config=config)()

        # Forces predicted by neural network.
        # Initialized with 0 for each engine.
        self.forces = [0.0 for _ in range(self.num_engines)]

        self.max_force = config.env.booster.engine.main.max_force
        # self.max_force = config.env.booster.engine.coldgas.max_force # TODO

        # Input data for neural network
        self.data = []

        # Fitness score
        self.score = 0.0

    def fetch_data(self):
        """Fetches data from booster that is fed into neural network."""
        self.data = [
            self.body.position.x,
            self.body.position.y,
            self.body.linearVelocity.x,
            self.body.linearVelocity.y,
            self.body.angularVelocity,
            self.body.angle,
        ]

    def mutate(self, model: object) -> None:
            """Mutates drone's neural network.

            TODO: Move to genetic optimizer. This is only for genetic optimization relevant.

            Args:
                model: The current best model.
            """
            self.model = copy.deepcopy(model)
            self.model.mutate_weights()

    def comp_score(self) -> None:
        """Computes current fitness score.

        Accumulates drone's linear velocity over one generation.
        This effectively computes the distance traveled by the
        drone over time divided by the simulation's step size.

        """
        if self.body.active:

            # Reward distance traveled.
            vel = self.body.linearVelocity
            time_step = 0.0167
            score = time_step * (vel.x**2 + vel.y**2) ** 0.5
            self.score += score

            # Reward distance to obstacles.
            # eta = 4.0
            # phi = 0.5
            # score = 1.0
            # for cb in self.callbacks:
            #     diff = cb.point - self.body.position
            #     dist = (diff.x**2 + diff.y**2) ** 0.5
            #     if dist < eta * self.collision_threshold:
            #         score = 0.0
            #         break
            # self.score += phi * score

    def detect_collision(self):
        """Detects collision with objects.
        We use the raycast information here and speak of a collision
        when an imaginary circle with the total diameter of the drone
        touches another object.
        """
        distance_booster_ground = 1
        if self.body.active:
            if distance_booster_ground < 0:
                self.body.active = False
                self.forces = self.num_engines * [0.0]

    def comp_action(self) -> None:
        """Computes next section of actions applied to engines.
        Next steps of action are computed by feeding obstacle data coming
        from ray casting to the drone's neural network which then returns
        a set of actions (forces) to be applied to the drone's engines.
        """
        if self.body.active:
            force_pred = self.model(self.data)
            self.forces = self.max_force * force_pred

    def apply_action(self) -> None:
        """Applies force to engines predicted by neural network.

        """
        if self.body.active:
            f_main, f_left, f_right = self.forces

            # Main engine
            f = self.body.GetWorldVector(localVector=b2Vec2(0.0, f_main))
            p = self.body.GetWorldPoint(localPoint=b2Vec2(0.0, -(0.5 * self.hull.height + self.engines.height)))
            self.body.ApplyForce(f, p, True)

            # Left cold gas thruster
            f = self.body.GetWorldVector(localVector=b2Vec2(f_left, 0.0))
            p = self.body.GetWorldPoint(localPoint=b2Vec2(-0.5 * self.hull.width, 0.5 * self.hull.height))
            self.body.ApplyForce(f, p, True)

            # Right cold gas thruster
            f = self.body.GetWorldVector(localVector=b2Vec2(-f_right, 0.0))
            p = self.body.GetWorldPoint(localPoint=b2Vec2(0.5 * self.hull.width, 0.5 * self.hull.height))
            self.body.ApplyForce(f, p, True)

    # def apply_action(
    #     self,
    #     force_merlin_engine: tuple = (0.0, 0.0),
    #     force_cold_gas_engine: tuple = ((0.0, 0.0), (0.0, 0.0)),
    # ):
    #     """
    #     NOTE: This is from the old implementation.


    #     Applies action coming from neural network.

    #     Network returns for merlin engines

    #         F'x, F'y in the range [0, 1] (sigmoid)

    #     The following transformation is used to avoid exceeding engine's max thrust

    #         F_x = F_max * F'_x / sqrt(2)
    #         F_y = F_max * F'_y / sqrt(2)

    #     Taking into account the engine's maximum deflection

    #         F_x = min(F_x_max, F_x) = min(sin(alpha), F_x)
    #         F_y = min(F_y_max, F_y) = min(cos(alpha), F_y)

    #     TODO: Shouldn't this be part of Booster class? Booster's networks
    #     computes set of actions and Booster's control system method executes
    #     actions.
    #     """
    #     # comp_actions() predicts which forces to apply
    #     # The predictions are a vector for each booster with F_x and F_y
    #     f_x_max = self.config.env.booster.engine.main.max_force
    #     f_y_max = self.config.env.booster.engine.main.max_force

    #     self.force_merlin_engine = [
    #         (random.uniform(-1, 1) * f_x_max, random.uniform(0, 1) * f_y_max)
    #         for _ in self.boosters
    #     ]  # some fake data

    #     f_max = self.config.env.booster.engine.cold_gas.max_force
    #     self.force_cold_gas_engine_left = [
    #         (random.random() * f_max, 0.0) for _ in self.boosters
    #     ]  # some fake data
    #     self.force_cold_gas_engine_right = [
    #         (-random.random() * f_max, 0.0) for _ in self.boosters
    #     ]  # some fake data

    #     for booster, force_merlin, force_cold_gas_left, force_cold_gas_right in zip(
    #         self.boosters,
    #         self.force_merlin_engine,
    #         self.force_cold_gas_engine_left,
    #         self.force_cold_gas_engine_right,
    #     ):

    #         #####################################
    #         # Apply force coming from main engine
    #         #####################################
    #         f = booster.body.GetWorldVector(
    #             localVector=force_merlin
    #         )  # Get the world coordinates of a vector given the local coordinates.

    #         local_point_merlin = b2Vec2(
    #             0.0, -(0.5 * booster.hull.height + booster.engines.height)
    #         )
    #         local_point_merlin = b2Vec2(
    #             0.0, -(0.5 * booster.hull.height + booster.engines.height)
    #         )

    #         p = booster.body.GetWorldPoint(localPoint=local_point_merlin)
    #         # Apply force f to point p of booster.
    #         booster.body.ApplyForce(f, p, True)

    #         ###########################################
    #         # Apply force coming from cold gas thruster
    #         ###########################################
    #         # Left
    #         local_point_cold_gas_left = b2Vec2(
    #             -0.5 * booster.hull.width, 0.5 * booster.hull.height
    #         )
    #         f = booster.body.GetWorldVector(
    #             localVector=force_cold_gas_left
    #         )  # Get the world coordinates of a vector given the local coordinates.
    #         p = booster.body.GetWorldPoint(
    #             localPoint=local_point_cold_gas_left
    #         )  # Get the world coordinates of a point given the local coordinates. Hence, p = booster.position + localPoint, with local coord. equals localPoint
    #         booster.body.ApplyForce(f, p, True)

    #         # Right
    #         local_point_cold_gas_right = b2Vec2(
    #             0.5 * booster.hull.width, 0.5 * booster.hull.height
    #         )
    #         f = booster.body.GetWorldVector(
    #             localVector=force_cold_gas_right
    #         )  # Get the world coordinates of a vector given the local coordinates.
    #         p = booster.body.GetWorldPoint(
    #             localPoint=local_point_cold_gas_right
    #         )  # Get the world coordinates of a point given the local coordinates. Hence, p = booster.position + localPoint, with local coord. equals localPoint
    #         booster.body.ApplyForce(f, p, True)

    def _print_booster_info(self) -> None:
        """Prints booster data."""
        print(f"{self.body.transform = }")
        print(f"{self.body.position = }")
        print(f"{self.body.angle = }")
        print(f"{self.body.localCenter = }")
        print(f"{self.body.worldCenter = }")
        print(f"{self.body.massData = }")
        print(f"{self.body.linearVelocity = }")
        print(f"{self.body.angularVelocity = }\n")