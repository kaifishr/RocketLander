"""Angle of object over time."""
import math

from Box2D.examples.framework import Framework, main
from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape, b2Vec2


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

        obj_vertices = [(2.0, 1.0), (-2.0, 1.0), (-2.0, -1.0), (2.0, -1.0)]

        # Create object
        pos1 = b2Vec2(0, 5.0) + (0, 0)
        velocity = (0.0, 0.0)
        self.obj1 = self.world.CreateDynamicBody(
            position=pos1,
            linearVelocity=velocity,
            angularVelocity=0.0
            * math.pi
            / 180,  # 180 deg per second, counterclockwise for positive numbers
            angle=170 * math.pi / 180.0,  # counterclockwise
            allowSleep=False,
        )

        # Upper part (central object)
        obj1_h = 1
        obj1_w = 1
        obj1_density = 1
        obj1_vertices = [(obj1_w * item[0], obj1_h * item[1]) for item in obj_vertices]
        obj1_shape = b2PolygonShape(vertices=obj1_vertices)
        obj1_fixture_def = b2FixtureDef(shape=obj1_shape, density=obj1_density)
        _ = self.obj1.CreateFixture(obj1_fixture_def)

    def Step(self, settings):
        super(TestObject, self).Step(settings)
        data = [
            self.obj1.position.x,
            self.obj1.position.y,
            self.obj1.linearVelocity.x,
            self.obj1.linearVelocity.y,
            self.obj1.angularVelocity,
            self.obj1.angle,
        ]
        # print(self.obj1.angularVelocity)
        print(self.obj1.transform.angle)
        # print(self.obj1.transform.angle, self.obj1.angle % (2.0 * math.pi))


if __name__ == "__main__":
    main(TestObject)
