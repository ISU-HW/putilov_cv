import pygame
import cv2
import numpy as np
import math
import sys


class ObstacleDetectorImageLike:
    def __init__(self, source):
        self.source = source

    def process_image_common(self, img):
        imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, thresh = cv2.threshold(imghsv[:, :, 1], 65, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=10)
        thresh = cv2.erode(thresh, None, iterations=0)
        return thresh


class ObstacleDetectorImage(ObstacleDetectorImageLike):
    def __init__(self, image_source):
        super().__init__(image_source)
        self.image = cv2.imread(self.source)

    def process_image(self):
        if self.image is None:
            raise ValueError("Изображение не найдено")
        return self.process_image_common(self.image)


class ObstacleDetectorCamera(ObstacleDetectorImageLike):
    def __init__(self, camera_source):
        super().__init__(camera_source)
        self.camera = cv2.VideoCapture(self.source + cv2.CAP_DSHOW)

    def process_image(self):
        if self.camera is None:
            raise ValueError("Не удалось получить доступ к видеопотоку")
        ret, img = self.camera.read()
        if img is None:
            raise ValueError("Не удалось получить изображение из видеопотока")
        return self.process_image_common(img)


class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 20
        self.speed_x = 0
        self.speed_y = 0

    def update_position(self, gravity, friction, bounce, thresh, width, height):
        self.speed_y += gravity
        new_x = self.x + self.speed_x
        new_y = self.y + self.speed_y
        if self.check_collision(thresh, new_x, new_y + self.radius):
            normal_x, normal_y = self.get_surface_normal(
                thresh, new_x, new_y + self.radius
            )
            velocity_along_normal = self.speed_x * normal_x + self.speed_y * normal_y
            velocity_along_tangent_x = self.speed_x - velocity_along_normal * normal_x
            velocity_along_tangent_y = self.speed_y - velocity_along_normal * normal_y
            velocity_along_tangent_x *= friction
            velocity_along_tangent_y *= friction
            self.speed_x = velocity_along_tangent_x
            self.speed_y = velocity_along_tangent_y
            self.x += self.speed_x
            self.y += self.speed_y
            self.speed_x = -velocity_along_normal * normal_x * bounce
            self.speed_y = -velocity_along_normal * normal_y * bounce
        else:
            self.x = new_x
            self.y = new_y
        if self.y + self.radius > height:
            self.y = height - self.radius
            self.speed_y = 0

    def get_surface_normal(self, thresh, x, y):
        if x < 0 or y < 0 or x >= thresh.shape[1] or y >= thresh.shape[0]:
            return 0, -1
        gx = cv2.Sobel(thresh, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(thresh, cv2.CV_64F, 0, 1, ksize=5)
        nx, ny = -gx[int(y), int(x)], -gy[int(y), int(x)]
        length = math.hypot(nx, ny)
        if length == 0:
            return 0, -1
        return nx / length, ny / length

    def check_collision(self, thresh, x, y):
        if x < 0 or y < 0 or x >= thresh.shape[1] or y >= thresh.shape[0]:
            return False
        return thresh[int(y), int(x)] == 255


class Game:
    def __init__(self, obstacle_detector):
        self.obstacle_detector = obstacle_detector
        self.height, self.width = self.obstacle_detector.image.shape[:2]
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Виртуальный шар")
        self.RED = (255, 0, 0)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def run(self):
        running = True
        clock = pygame.time.Clock()
        ball = Ball(0, 20)

        while running:
            running = self.handle_events()

            try:
                processed_img = self.obstacle_detector.process_image()
            except ValueError as e:
                print("Ошибка:", e)
                continue

            if processed_img is not None:
                resized_image = cv2.resize(self.obstacle_detector.image, (self.width, self.height))
                pygame.surfarray.blit_array(
                    self.screen,
                    cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                )

                ball.update_position(
                    gravity=0.3,
                    friction=0.98,
                    bounce=0.8,
                    thresh=processed_img,
                    width=self.width,
                    height=self.height,
                )

                self.screen.fill((255, 255, 255))
                pygame.draw.circle(
                    self.screen, self.RED, (int(ball.x), int(ball.y)), ball.radius
                )

                pygame.display.flip()
                cv2.waitKey(1)

            clock.tick(60)

        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    obstacle_detector = ObstacleDetectorImage("doska.jpg")
    game = Game(obstacle_detector)
    game.run()



if __name__ == "__main__":
    obstacle_detector = ObstacleDetectorImage("doska.jpg")
    game = Game(obstacle_detector)
    game.run()
