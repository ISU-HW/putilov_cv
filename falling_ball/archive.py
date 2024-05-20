from multiprocessing import process
import pygame as g
import numpy as np


class GravitationModule:
    def __init__(self, gravity=9.8):
        self.gravity = gravity

    def apply_gravity(self):
        self.velocity[1] += self.gravity


class CollisionModule:
    def __init__(self, acceleration=1.5, restitution=0.8) -> None:
        self.acceleration = acceleration
        self.restitution = restitution


class Ball(GravitationModule):
    def __init__(self, x, y, radius, color):
        super().__init__(gravity=0.98)
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.velocity = np.array([0.0, 0.0])

    def draw(self, screen):
        g.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def update(self):
        self.x += self.velocity[0]
        self.y += self.velocity[1]

    def process_collision(self, collision, collided):

        self.velocity[0] -= (
            2
            * collision["dot_product"]
            * collision["normalized_direction_vector"][0]
            * collided.restitution
        )
        self.velocity[1] -= (
            2
            * collision["dot_product"]
            * collision["normalized_direction_vector"][1]
            * collided.restitution
        )

        self.velocity[0] += (
            collision["normalized_direction_vector"][0] * collided.acceleration
        )
        self.velocity[1] += (
            collision["normalized_direction_vector"][1] * collided.acceleration
        )


class Obstacle(CollisionModule):
    def __init__(self, x, y, width, height, color, rotation=0):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.rotation = rotation

    @property
    def collision_box(self):
        s = g.Surface((self.width, self.height), g.SRCALPHA)
        s.fill(self.color)
        rotated_surface = g.transform.rotate(s, self.rotation)
        return rotated_surface

    def draw(self, screen):
        screen.blit(
            self.collision_box,
            (
                self.x - self.collision_box.get_width() / 2,
                self.y - self.collision_box.get_height() / 2,
            ),
        )


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

if __name__ == "__main__":
    running = True
    g.init()
    screen = g.display.set_mode((640, 500))
    g.display.set_caption("Ball")
    clock = g.time.Clock()
    screen.fill(BLACK)
    ball1 = Ball(320, 50, 15, WHITE)
    obstacle1 = Obstacle(350, 200, 200, 20, RED, 30)
    obstacle2 = Obstacle(170, 300, 200, 20, BLUE, -45)
    obstacle3 = Obstacle(320, 500, 400, 150, GREEN)
    obstacle1.draw(screen)
    obstacle2.draw(screen)
    obstacle3.draw(screen)
    while running:
        for event in g.event.get():
            if event.type == g.QUIT:
                running = False

        ball1.apply_gravity()
        ball1.update()

        if ball1.y > screen.get_height():
            ball1.x = 320
            ball1.y = 50
            ball1.velocity = np.array([0.0, 0.0])

        obstacles = [obstacle1, obstacle2, obstacle3]

        for obstacle in obstacles:
            obstacle_mask = g.mask.from_surface(obstacle.collision_box)
            collision = obstacle_mask.overlap(ball1)
            if collision:
                ball1.process_collision(collision, obstacle)

        screen.fill(BLACK)

        ball1.draw(screen)
        obstacle1.draw(screen)
        obstacle2.draw(screen)
        obstacle3.draw(screen)

        g.display.flip()
        g.time.Clock().tick(20)
    g.quit()
