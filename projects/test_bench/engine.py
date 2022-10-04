"""Script to visualize engine / force activities in pybox2d."""
import random

from Box2D.examples.framework import Framework, main
from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape, b2Vec2, b2Color


class TestObject(Framework):

    def __init__(self):
        super(TestObject, self).__init__()

        self.axisScale = 1.0

        # World
        self.world.gravity = (0.0, -10.0)
        self.world.CreateStaticBody(
            shapes=[
                    b2EdgeShape(vertices=[(-10, 0), (10, 0)]),
                    b2EdgeShape(vertices=[(-10, 0), (-20, 20)]),
                    b2EdgeShape(vertices=[(10, 0), (20, 20)]),
                ]
        )

        obj_vertices = [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]

        # Create object
        height, width = 6, 4
        position = b2Vec2(12, 20)
        linear_velocity = (0.0, -10.0)
        self.obj = self.world.CreateDynamicBody(position=position, linearVelocity=linear_velocity, allowSleep=False)

        # Engine
        density = 1
        vertices = [(width * item[0], height * item[1]) for item in obj_vertices]
        shape = b2PolygonShape(vertices=vertices)
        fixture_def = b2FixtureDef(shape=shape, density=density)
        self.obj.CreateFixture(fixture_def)

        print(f"{self.obj = }")
        self.max_force = 100.0

        self.width = width

    def force_display(self, line_width=0.1):

        # Compute random force
        force_x = self.max_force * random.gauss(0, 1)
        force_y = self.max_force * random.gauss(0, 1)

        self.obj.ApplyForceToCenter(
            (
                force_x,
                force_y,
            ),
            True,
        )

        # Line
        alpha = 0.02
        force_start = (0.0, 0.0)
        self.line_color = (0, 1, 0)
        p1 = self.obj.GetWorldPoint(force_start)

        force_length = force_x   # should be linear function of force
        force_direction = (alpha * force_length, 0.0)
        p2 = p1 + self.obj.GetWorldVector(force_direction)
        self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

        force_length = force_y   # should be linear function of force
        force_direction = (0.0, alpha * force_length)
        p2 = p1 + self.obj.GetWorldVector(force_direction)
        self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))


    def Step(self, settings):
        super(TestObject, self).Step(settings)
        self.force_display()


if __name__ == "__main__":
    main(TestObject)