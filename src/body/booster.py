"""Defines rocket booster in Box2d."""
import copy
import math

import numpy
import numpy as np

from Box2D import b2FixtureDef, b2PolygonShape, b2Filter, b2Vec2
from Box2D.Box2D import b2World, b2Body

from src.utils import Config
from src.body.model import ModelLoader

# class ControlSystem:
# 
#     def __init__(self):
# 
#         model = Model()
#         data_processing = DataProcessing()


class Booster2D:  # BoosterBody
    """Rocket booster class.

    Rocket booster consists of three main parts:
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

        # Parameters of booster's hull.
        density = 1.0   # Average density of hull. (~465 kg / m)
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
        density = 18.0 # Average density of engines, (~2820 kg / m), times 3 to account for real merlin size
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

        init = config.env.booster.init

        self.init_position = b2Vec2(
            init.position.x,
            init.position.y,
        )
        self.init_linear_velocity = b2Vec2(
            init.linear_velocity.x,
            init.linear_velocity.y,
        )

        self.init_angular_velocity = self._deg_to_rad(init.angular_velocity)
        self.init_angle = self._deg_to_rad(init.angle)

        self.fixed_rotation = config.env.booster.fixed_rotation

        self.body = self.world.CreateDynamicBody(
            bullet=True,
            allowSleep=False,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle=self.init_angle,
            fixedRotation=self.fixed_rotation,
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

    @staticmethod
    def _deg_to_rad(deg: float) -> float:
        """Converts from degrees to radians.
        
        Args:
            float: Number of degrees.
            
        Returns:
            Radians.
        """
        rad = deg * math.pi / 180.0
        return rad

class Booster(Booster2D):
    """Booster class holds PyBox2D booster object as well
    as the booster logic"""

    num_engines = 3
    num_dims = 2

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes Booster class."""
        super().__init__(world=world, config=config)

        self.model = ModelLoader(config=config)()

        # Forces predicted by neural network.
        # self.predictions = [0.0 for _ in range(self.num_dims * self.num_engines)]
        self.predictions = np.zeros(shape=(self.num_dims * self.num_engines))
        self.max_force_main = config.env.booster.engine.main.max_force
        self.max_angle_main = config.env.booster.engine.main.max_angle
        self.max_force_cold_gas = config.env.booster.engine.cold_gas.max_force
        self.max_angle_cold_gas = config.env.booster.engine.cold_gas.max_angle

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
            self.body.transform.angle, 
        ]

    def _pre_process(self, data: numpy.ndarray) -> numpy.ndarray:
        """Applies pre-processing to fetched data.

        TODO: Use online method to compute running mean and standard deviation
              for data normalization.

        Normalizes fetched data.

        Args:
            data: Array holding fetched data.

        Returns:
            Array with normalized data.
        """
        pos_x, pos_y = data[0], data[1]
        vel_x, vel_y = data[2], data[3]
        angular_vel = data[4]
        angle = data[5]

        # Position
        pos_x_min, pos_x_max = -200.0, 200.0
        pos_y_min, pos_y_max = -5.0, 300.0
        pos_x = 2.0 * (pos_x - pos_x_min) / (pos_x_max - pos_x_min) - 1.0
        pos_y = 2.0 * (pos_y - pos_y_min) / (pos_y_max - pos_y_min) - 1.0
        data[0], data[1] = pos_x, pos_y

        # Velocity
        vel_x_min, vel_x_max = -50.0, 50.0
        vel_y_min, vel_y_max = 0.0, 100.0
        vel_x = 2.0 * (vel_x - vel_x_min) / (vel_x_max - vel_x_min) - 1.0
        vel_y = 2.0 * (vel_y - vel_y_min) / (vel_y_max - vel_y_min) - 1.0
        data[2], data[3] = vel_x, vel_y

        # Angular velocity
        angular_vel_min, angular_vel_max = -0.5 * math.pi, 0.5 * math.pi     # [pi rad / s]
        angular_vel = 2.0 * (angular_vel - angular_vel_min) / (angular_vel_max - angular_vel_min) - 1.0
        data[4] = angular_vel

        # Angle
        angle_min, angle_max = -0.5 * math.pi, 0.5 * math.pi     # [pi rad]
        angle = 2.0 * (angle - angle_min) / (angle_max - angle_min) - 1.0
        data[5] = angle

        return data

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

        Accumulates defined rewards for booster.
        """
        if self.body.active:

            pos_x, pos_y = self.body.position
            vel_x, vel_y = self.body.linearVelocity

            # Reward proximity to landing pad if sink rate is negative.
            if vel_y <= 0.0:
                pos_pad = self.config.env.landing_pad.position
                # eta = 1.0 / 60.0
                pos_y -= 0.5 * self.hull.height - self.legs.y_ground #+ eta
                dist = ((pos_pad.x - pos_x) ** 2 + (pos_pad.y - pos_y) ** 2) ** 0.5
                eta = 1.0
                reward = 1.0 / (1.0 + eta * dist)
                # reward = 1.0 / (1.0 + eta * dist + abs(vel_y))
                # reward = 1.0 / (1.0 + (eta * dist + abs(vel_y))**2)
                # reward = math.exp(-(eta * dist + abs(vel_y)) ** 2)
                #print("proximity", reward)
                self.score += reward

                # Reward for soft touchdown 
                # vel_x, vel_y = self.body.linearVelocity
                # vel = (vel_x ** 2 + vel_y ** 2) ** 0.5
                # reward = 1.0 / (1.0 + (distance_booster_pad * vel))

                # # Reward for verticality 
                # eta = 1.0
                # reward = 1.0 / (1.0 + eta * abs(self.body.transform.angle))
                # #print("angle", reward, self.body.angle)
                # self.score += reward

                # # Reward for low angular velocity 
                # eta = 1.0
                # reward = 1.0 / (1.0 + eta * abs(self.body.angularVelocity))
                # # print("angular_velocity", reward, self.body.angularVelocity)
                # self.score += reward

            else:
                self.body.active = False
                self.predictions = np.zeros(shape=(self.num_dims * self.num_engines))

    def detect_impact(self):
        """Detects impact with ground.

        TODO: Move to environment?
        TODO: Move threshold computation to __init__()

        Deactivates booster in case of impact.

        An impact has occurred when a booster is one
        length unit above the ground at a higher than
        defined velocity. 

        For contact calculation a circle with radius R = (a^2+b^2)^0.5 (contact
        threshold) around the rocket is assumed. The rocket has 'contact' if the
        vertical distance from the center of mass to the ground is smaller than R.
        """
        #print(self.body.position, self.body.linearVelocity)
        if self.body.active:
            v_max_x = self.config.env.landing.v_max.x
            v_max_y = self.config.env.landing.v_max.y

            # Compute distance from center of mass to ground
            eta = 1.0 / 60.0  # 0.0167 # ~ 1.0 / 60.0
            a = self.legs.x_ground_high
            b = 0.5 * self.hull.height - self.legs.y_ground + eta
            contact_threshold = (a**2 + b**2) ** 0.5 + 2.0

            # Check for impact
            # print(self.body.position.y, self.body.linearVelocity, contact_threshold)
            if self.body.position.y < contact_threshold:    # TODO: Use circle around booster coming from center at center of mass.
                #print("Contact")
                vel_x, vel_y = self.body.linearVelocity
                if (vel_y < v_max_y) or (abs(vel_x) > v_max_x):
                    #print("Impact")
                    self.body.active = False
                    self.predictions = np.zeros(shape=(self.num_dims * self.num_engines))

    def detect_excess_stress(self) -> None:
        """Detects excess stress on booster.

        Deactivates booster if stress limits are being exceeded.
        """
        stress = self.config.env.booster.stress

        max_angle = self._deg_to_rad(stress.max_angle)
        max_angular_velocity = self._deg_to_rad(stress.max_angular_velocity)

        # Angle
        if abs(self.body.transform.angle) > max_angle:
            self.body.active = False
            self.predictions = np.zeros(shape=(self.num_dims * self.num_engines))
            return
        # Angular velocity
        elif abs(self.body.angularVelocity) > max_angular_velocity:
            self.body.active = False
            self.predictions = np.zeros(shape=(self.num_dims * self.num_engines))
            return

    def comp_action(self) -> None:
        """Computes next section of actions applied to engines.
        Next steps of action are computed by feeding obstacle data coming
        from ray casting to the drone's neural network which then returns
        a set of actions (forces) to be applied to the drone's engines.
        """
        if self.body.active:

            # Pre-processing
            # data = self._pre_process(self.data)
            data = self.data

            # Raw network predictions
            pred = self.model(data)

            # Post-processing
            self.predictions = self._post_process(pred)
 
    def _post_process(self, pred: numpy.ndarray) -> tuple:
        """Applies post-processing to raw network output.

        Network predicts for each engine level of thrust and angle. Predictions are
        between [0, 1]. From these predictions as well as from the maximum power of
        the engines and gimbal angle, the force components f_x and f_y are computed
        for each engine. Thus,

        pred = [
            p_main, 
            phi_main, 
            p_left, 
            phi_left, 
            p_right, 
            phi_right
        ]

        is transformed to

        predictions = [
            f_x_main,
            f_y_main,
            f_x_left,
            f_y_left,
            f_x_right,
            f_y_right
        ]

        Args:
            pred: Raw network predictions.

        Returns:
            Array of force components f_x and f_y for each engine.
        """
        p_main, phi_main, p_left, phi_left, p_right, phi_right = pred

        # Main engine
        phi = 2.0 * phi_main - 1.0  # [-1, 1]
        arg = phi * self.max_angle_main * math.pi / 180.0
        force = p_main * self.max_force_main
        f_main_x = math.sin(arg) * force
        f_main_y = math.cos(arg) * force

        # Left engine
        phi = 2.0 * phi_left - 1.0  # [-1, 1]
        arg = phi * self.max_angle_cold_gas * math.pi / 180.0
        force = p_left * self.max_force_cold_gas
        f_left_x = math.cos(arg) * force
        f_left_y = math.sin(arg) * force

        # Right engine
        phi = 2.0 * phi_right - 1.0  # [-1, 1]
        arg = phi * self.max_angle_cold_gas * math.pi / 180.0
        force = p_right * self.max_force_cold_gas
        f_right_x = math.cos(arg) * force
        f_right_y = math.sin(arg) * force

        return f_main_x, f_main_y, f_left_x, f_left_y, f_right_x, f_right_y

    def apply_action(self) -> None:
        """Applies force to engines predicted by neural network."""
        if self.body.active:

            f_main_x, f_main_y, f_left_x, f_left_y, f_right_x, f_right_y = self.predictions

            # Main engine x-component
            f = self.body.GetWorldVector(localVector=b2Vec2(f_main_x, f_main_y))
            p = self.body.GetWorldPoint(
                localPoint=b2Vec2(0.0, -(0.5 * self.hull.height + self.engines.height))
            )
            self.body.ApplyForce(f, p, True)

            # Left cold gas thruster
            f = self.body.GetWorldVector(localVector=b2Vec2(f_left_x, f_left_y))
            p = self.body.GetWorldPoint(
                localPoint=b2Vec2(-0.5 * self.hull.width, 0.5 * self.hull.height)
            )
            self.body.ApplyForce(f, p, True)

            # Right cold gas thruster
            f = self.body.GetWorldVector(localVector=b2Vec2(-f_right_x, f_right_y))
            p = self.body.GetWorldPoint(
                localPoint=b2Vec2(0.5 * self.hull.width, 0.5 * self.hull.height)
            )
            self.body.ApplyForce(f, p, True)

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
