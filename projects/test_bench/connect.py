"""Test lab for pybox2d
"""
import random

from Box2D.examples.framework import Framework, main
from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape, b2Vec2


# Use this as template for rocket booster and landing pad.
# Move code to respective files.
class Box:
    """Template class for objects.

    Example usage:

    self.boxes = list()
    for _ in range(20):
        self.boxes.append(Box()(world=self.world))
    """

    def __init__(self):
        """"""
        self.engines_prototype = [(1.0, -1.0), (-1.0, -1.0), (-1.0, -2.0), (1.0, -2.0)]

    @staticmethod
    def _pos_rand(p_min, p_max):
        return (p_max - p_min) * random.random() + p_min

    def __call__(self, world) -> None:
        """Adds a box to world."""

        # Position
        min_pos_x = -20
        max_pos_x = 20
        min_pos_y = 80
        max_pos_y = 120

        init_pos_x = self._pos_rand(min_pos_x, max_pos_x)
        init_pos_y = self._pos_rand(min_pos_y, max_pos_y)

        init_position = (init_pos_x, init_pos_y)

        # Velocity
        init_velocity = (0.0, 0.0)

        # Add dynamic body to the world
        box = world.CreateDynamicBody(
            bullet=True,
            position=init_position,
            linearVelocity=init_velocity,
        )

        # Add properties to dynamic body
        vertices = [
            (5 * random.random() * x, 5 * random.random() * y)
            for (x, y) in self.engines_prototype
        ]
        density = 100 * random.random() + 1

        box_fixture = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices), density=density
        )
        _ = box.CreateFixture(box_fixture)

        return box


class TestObject(Framework):
    def __init__(self):
        super(TestObject, self).__init__()

        # World
        self.world.gravity = (0.0, 0.0)
        self.world.CreateStaticBody(
            shapes=[
                b2EdgeShape(vertices=[(-10, 0), (10, 0)]),
                b2EdgeShape(vertices=[(-10, 0), (-20, 20)]),
                b2EdgeShape(vertices=[(10, 0), (20, 20)]),
            ]
        )

        obj_vertices = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]

        # Create object
        pos1 = b2Vec2(12, 20) + (0, 0)
        velocity = (0.0, -10.0)
        obj1 = self.world.CreateDynamicBody(
            position=pos1, linearVelocity=velocity, allowSleep=False
        )

        # Upper part (central object)
        obj1_h = 8
        obj1_w = 1
        obj1_density = 1
        obj1_vertices = [(obj1_w * item[0], obj1_h * item[1]) for item in obj_vertices]
        obj1_shape = b2PolygonShape(vertices=obj1_vertices)
        obj1_fixture_def = b2FixtureDef(shape=obj1_shape, density=obj1_density)
        _ = obj1.CreateFixture(obj1_fixture_def)

        # Left leg
        leg_density = 1.0
        left_leg = b2PolygonShape(
            vertices=((-1.0, -obj1_h), (-4, -11), (-4.1, -11), (-1.1, -obj1_h))
        )
        left_leg_fixture_ = b2FixtureDef(shape=left_leg, density=leg_density)
        _ = obj1.CreateFixture(left_leg_fixture_)

        # Left leg
        right_leg = b2PolygonShape(
            vertices=((1.0, -obj1_h), (4, -11), (4.1, -11), (1.1, -obj1_h))
        )
        right_leg_fixture_ = b2FixtureDef(shape=right_leg, density=leg_density)
        _ = obj1.CreateFixture(right_leg_fixture_)

        # Engines
        obj2_h = 0.75
        obj2_w = 1
        obj2_density = 100
        obj2_vertices = [
            (obj2_w * item[0], obj2_h * item[1] - obj1_h - obj2_h)
            for item in obj_vertices
        ]
        obj2_shape = b2PolygonShape(vertices=obj2_vertices)
        # obj2_shape = b2PolygonShape(vertices=((1.0), (), (), ()))
        obj2_fixture_def = b2FixtureDef(shape=obj2_shape, density=obj2_density)
        _ = obj1.CreateFixture(obj2_fixture_def)

        print(f"{obj1 = }")

    def Step(self, settings):
        super(TestObject, self).Step(settings)


if __name__ == "__main__":
    main(TestObject)
