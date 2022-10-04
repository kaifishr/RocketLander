"""Defines rocket booster in Box2d."""
import math
import random
import numpy as np

from Box2D import b2FixtureDef, b2PolygonShape, b2Filter, b2Vec2
from Box2D.Box2D import b2World, b2Body

from config import Config


class Booster2D:
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
            self.vertices = [(self.width * x, self.height * y) for (x, y) in self.vertices]
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
        height = 1.0
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
                r_1 = mount_point + b2Vec2(- 0.5 * self.width_min, 0.0)
                r_2 = mount_point + b2Vec2(- 0.5 * self.width_max, -self.height)
                r_3 = mount_point + b2Vec2(0.5 * self.width_max, -self.height)
                return r_0, r_1, r_2, r_3

            mount_points = [
                b2Vec2(0.0, -0.5 * self.hull.height),
                b2Vec2(-(1.0 / 3.0) * self.hull.width, -0.5 * self.hull.height),
                b2Vec2((1.0 / 3.0) * self.hull.width, -0.5 * self.hull.height)
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
            """Adds landing legs to booster.
            """

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
                filter=b2Filter(groupIndex=-1)     # no interactions with any part of other boosters
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
                filter=b2Filter(groupIndex=-1)     # no interactions with any part of other boosters
            )

            right_leg_fixture = self.body.CreateFixture(right_leg_fixture)

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes Booster2D class."""

        self.config = config
        self.world = world

        self.init_position = b2Vec2(
            config.env.booster.init_pos.x,
            config.env.booster.init_pos.y,
        )
        self.init_linear_velocity = b2Vec2(
            config.env.booster.init_linear_vel.x,
            config.env.booster.init_linear_vel.y,
        )
        self.init_angular_velocity = config.env.booster.init_angular_vel
        self.init_angle = (config.env.booster.init_angle * math.pi) / 180.0

        self.body = self.world.CreateDynamicBody(
            bullet=True,
            allowSleep=False,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle = self.init_angle,
        )

        self.hull = self.Hull(
            body=self.body
        )

        self.engines = self.Engines(
            body=self.body,
            hull=self.hull,
        )

        self.legs = self.LandingLegs(
            body=self.body,
            hull=self.hull,
        )

    def reset(self, noise: bool = False) -> None:
        """Resets booster to initial position and velocity.
        """
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

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes Booster class."""
        super().__init__(world=world, config=config)
        # self.model = Model(dims_in, dims_out)

    def control_system(self, data):
        """The booster's brain.

        Data receives data and predicts new forces.

        TODO: Move this method to Booster() class. This class will be renamed Booster() -> Booster_()
        """
        # Toy predictor
        weights = np.random.normal(size=(2, 6))
        bias = np.random.normal(size=(2, ))
        pred = np.matmul(np.array(weights), data) + bias
        return pred