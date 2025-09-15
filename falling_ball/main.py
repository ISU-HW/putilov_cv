import cv2
import numpy as np
import pygame
import math
import time
import argparse
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class VectorPlatform:
    def __init__(
        self, contour: np.ndarray, color: Tuple[int, int, int] = (100, 100, 255)
    ):
        self.contour = contour
        self.simplified_contour = self._simplify_contour(contour)
        self.bbox = cv2.boundingRect(contour)
        self.x, self.y, self.w, self.h = self.bbox
        self.segments = self._create_segments()
        self.color = color

    def _simplify_contour(
        self, contour: np.ndarray, epsilon_factor: float = 0.02
    ) -> np.ndarray:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)

    def _create_segments(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        segments = []
        points = self.simplified_contour.reshape(-1, 2)

        for i in range(len(points)):
            start_point = tuple(points[i])
            end_point = tuple(points[(i + 1) % len(points)])
            segments.append((start_point, end_point))

        return segments

    def get_collision_point_and_normal(
        self, ball_x: float, ball_y: float, ball_radius: float
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        min_dist = float("inf")
        collision_point = None
        collision_normal = None

        for (x1, y1), (x2, y2) in self.segments:

            A = ball_x - x1
            B = ball_y - y1
            C = x2 - x1
            D = y2 - y1

            dot = A * C + B * D
            len_sq = C * C + D * D

            if len_sq == 0:

                closest_x, closest_y = x1, y1
            else:
                param = dot / len_sq

                if param < 0:
                    closest_x, closest_y = x1, y1
                elif param > 1:
                    closest_x, closest_y = x2, y2
                else:
                    closest_x = x1 + param * C
                    closest_y = y1 + param * D

            dist = math.sqrt((ball_x - closest_x) ** 2 + (ball_y - closest_y) ** 2)

            if dist < min_dist and dist <= ball_radius:
                min_dist = dist
                collision_point = (closest_x, closest_y)

                if len_sq > 0:

                    normal_x = -(y2 - y1) / math.sqrt(len_sq)
                    normal_y = (x2 - x1) / math.sqrt(len_sq)

                    to_ball_x = ball_x - closest_x
                    to_ball_y = ball_y - closest_y
                    if to_ball_x * normal_x + to_ball_y * normal_y < 0:
                        normal_x = -normal_x
                        normal_y = -normal_y

                    collision_normal = (normal_x, normal_y)

        if collision_point and collision_normal:
            return collision_point, collision_normal
        return None

    def contains_point(self, x: float, y: float) -> bool:
        point = (int(x), int(y))
        return cv2.pointPolygonTest(self.contour, point, False) >= 0


class Ball:
    def __init__(self, x: float, y: float, radius: float = 8):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.radius = radius
        self.gravity = 0.4
        self.friction = 0.95
        self.bounce = 0.6
        self.trail = []
        self.max_trail_length = 20

    def update(
        self, platforms: List[VectorPlatform], screen_width: int, screen_height: int
    ):
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

        self.vy += self.gravity

        old_x, old_y = self.x, self.y

        self.x += self.vx
        self.y += self.vy

        for platform in platforms:
            collision_result = platform.get_collision_point_and_normal(
                self.x, self.y, self.radius
            )

            if collision_result:
                collision_point, normal = collision_result

                overlap_distance = self.radius - math.sqrt(
                    (self.x - collision_point[0]) ** 2
                    + (self.y - collision_point[1]) ** 2
                )

                if overlap_distance > 0:
                    self.x += normal[0] * overlap_distance
                    self.y += normal[1] * overlap_distance

                dot_product = self.vx * normal[0] + self.vy * normal[1]
                if dot_product < 0:
                    self.vx -= 2 * dot_product * normal[0] * self.bounce
                    self.vy -= 2 * dot_product * normal[1] * self.bounce

                    self.vx *= self.friction
                    self.vy *= self.friction

        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * self.bounce
        elif self.x + self.radius > screen_width:
            self.x = screen_width - self.radius
            self.vx = -self.vx * self.bounce

        if self.y + self.radius > screen_height:
            self.y = screen_height - self.radius
            self.vy = -self.vy * self.bounce
            self.vx *= self.friction


class AdvancedPlatformDetector:
    def __init__(self, min_area: int = 300):
        self.min_area = min_area
        self.last_platforms = []
        self.background_model = None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filled = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_fill)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        dilated = cv2.dilate(filled, kernel_dilate, iterations=1)

        return dilated

    def _extract_platform_color(
        self, original_image: np.ndarray, contour: np.ndarray
    ) -> Tuple[int, int, int]:
        try:

            x, y, w, h = cv2.boundingRect(contour)

            roi = original_image[y : y + h, x : x + w]

            mask = np.zeros((h, w), dtype=np.uint8)

            shifted_contour = contour - [x, y]
            cv2.fillPoly(mask, [shifted_contour], 255)

            if len(roi.shape) == 3:

                masked_pixels = roi[mask > 0]

                if len(masked_pixels) > 0:

                    pixels = masked_pixels.reshape(-1, 3).astype(np.float32)

                    criteria = (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        10,
                        1.0,
                    )
                    _, labels, centers = cv2.kmeans(
                        pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
                    )

                    unique, counts = np.unique(labels, return_counts=True)
                    dominant_cluster = unique[np.argmax(counts)]
                    dominant_color = centers[dominant_cluster]

                    return (
                        int(dominant_color[2]),
                        int(dominant_color[1]),
                        int(dominant_color[0]),
                    )
                else:

                    mean_color = cv2.mean(roi)
                    return (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))
            else:

                masked_pixels = roi[mask > 0]
                if len(masked_pixels) > 0:
                    mean_gray = int(np.mean(masked_pixels))
                    return (mean_gray, mean_gray, mean_gray)
                else:
                    return (128, 128, 128)

        except Exception as e:
            print(f"Ошибка извлечения цвета: {e}")

            return (100, 150, 255)

    def _merge_nearby_contours(
        self, contours: List[np.ndarray], merge_distance: float = 15.0
    ) -> List[np.ndarray]:
        if len(contours) <= 1:
            return contours

        merged = []
        used = set()

        for i, contour1 in enumerate(contours):
            if i in used:
                continue

            group = [contour1]
            rect1 = cv2.boundingRect(contour1)

            for j, contour2 in enumerate(contours):
                if j <= i or j in used:
                    continue

                rect2 = cv2.boundingRect(contour2)

                distance = self._rect_distance(rect1, rect2)
                if distance < merge_distance:
                    group.append(contour2)
                    used.add(j)

            if len(group) == 1:
                merged.append(group[0])
            else:

                all_points = np.vstack(group)
                hull = cv2.convexHull(all_points)
                merged.append(hull)

            used.add(i)

        return merged

    def _rect_distance(
        self, rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]
    ) -> float:
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)

        return math.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )

    def detect_platforms_from_image(
        self, image_path: str, debug: bool = False
    ) -> List[VectorPlatform]:
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                return []
            return self.detect_platforms_from_frame(img, debug=debug)
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")
            return []

    def detect_platforms_from_frame(
        self, frame: np.ndarray, debug: bool = False
    ) -> List[VectorPlatform]:
        try:
            original = frame.copy()

            processed = self._preprocess_image(frame)

            contours, _ = cv2.findContours(
                processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if debug:
                self._show_debug_stages(original, processed, contours)

            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:

                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:

                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        if circularity < 0.9:
                            filtered_contours.append(contour)

            merged_contours = self._merge_nearby_contours(filtered_contours)

            platforms = []
            for contour in merged_contours:
                color = self._extract_platform_color(original, contour)
                platforms.append(VectorPlatform(contour, color))

            if platforms:
                platforms.sort(key=lambda p: p.y)
                self.last_platforms = platforms
                print(f"Найдено {len(platforms)} валидных площадок")
                return platforms
            else:
                if self.last_platforms:
                    print("Новые площадки не найдены, используем предыдущие")
                    return self.last_platforms
                else:
                    print("Площадки не обнаружены")
                    return []

        except Exception as e:
            print(f"Ошибка при детекции площадок: {e}")
            return self.last_platforms if self.last_platforms else []

    def _show_debug_stages(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        contours: list,
    ):
        try:
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 3, 1)
            if len(original.shape) == 3:
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(original, cmap="gray")
            plt.title("Оригинальное изображение")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(processed, cmap="gray")
            plt.title("Предобработка")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            contour_img = original.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            if len(contour_img.shape) == 3:
                plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(contour_img, cmap="gray")
            plt.title(f"Все контуры ({len(contours)})")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            filtered_img = original.copy()
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        if circularity < 0.9:
                            valid_contours.append(contour)

            cv2.drawContours(filtered_img, valid_contours, -1, (255, 0, 0), 3)
            if len(filtered_img.shape) == 3:
                plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(filtered_img, cmap="gray")
            plt.title(f"Валидные площадки ({len(valid_contours)})")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            merged_contours = self._merge_nearby_contours(valid_contours)
            merged_img = original.copy()
            cv2.drawContours(merged_img, merged_contours, -1, (0, 0, 255), 3)
            if len(merged_img.shape) == 3:
                plt.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(merged_img, cmap="gray")
            plt.title(f"После объединения ({len(merged_contours)})")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Ошибка при показе debug информации: {e}")


class DrawingPlatform:
    def __init__(self):
        self.drawing = False
        self.current_points = []
        self.completed_contours = []

    def start_drawing(self, x: int, y: int):
        self.drawing = True
        self.current_points = [(x, y)]

    def add_point(self, x: int, y: int):
        if self.drawing:
            if (
                not self.current_points
                or math.sqrt(
                    (x - self.current_points[-1][0]) ** 2
                    + (y - self.current_points[-1][1]) ** 2
                )
                > 5
            ):
                self.current_points.append((x, y))

    def finish_drawing(self) -> Optional[VectorPlatform]:
        if self.drawing and len(self.current_points) > 2:
            contour = np.array(
                [[point] for point in self.current_points], dtype=np.int32
            )

            platform = VectorPlatform(contour, (255, 165, 0))
            self.completed_contours.append(contour)
            self.drawing = False
            self.current_points = []
            return platform

        self.drawing = False
        self.current_points = []
        return None

    def cancel_drawing(self):
        self.drawing = False
        self.current_points = []


class CameraManager:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False

    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Не удалось открыть камеру {self.camera_id}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            self.is_opened = True
            print(f"Камера {self.camera_id} успешно инициализирована")
            return True

        except Exception as e:
            print(f"Ошибка инициализации камеры: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_opened or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False


class Game:
    def __init__(
        self,
        mode: str = "test",
        camera_id: int = 0,
        screen_width: int = 1920,
        screen_height: int = 1080,
        debug: bool = False,
    ):
        self.mode = mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.debug = debug
        self.fullscreen = True

        pygame.init()
        self.screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption(f"Векторная симуляция шарика - Режим: {mode}")
        self.clock = pygame.time.Clock()

        self.platform_detector = AdvancedPlatformDetector()
        self.drawing_platform = DrawingPlatform() if mode == "test" else None

        if mode == "live":
            self.camera = CameraManager(camera_id)
            if not self.camera.initialize():
                print("Переключение в тестовый режим из-за проблем с камерой")
                self.mode = "test"
                self.drawing_platform = DrawingPlatform()
                self.platforms = self.platform_detector.detect_platforms_from_image(
                    "doska.jpg", debug=self.debug
                )
            else:
                self.platforms = []
                self.update_platforms_from_camera()
        else:
            self.camera = None
            self.platforms = self.platform_detector.detect_platforms_from_image(
                "doska.jpg", debug=self.debug
            )

        self.ball = Ball(100, 50)

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 50, 50)
        self.GREEN = (50, 255, 50)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)

        self.running = True
        self.last_platform_update = 0
        self.platform_update_interval = 0.1
        self.show_debug = False

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height), pygame.FULLSCREEN
            )
        else:

            self.screen = pygame.display.set_mode((640, 480))
        print(f"Переключен в {'полноэкранный' if self.fullscreen else 'оконный'} режим")

    def update_platforms_from_camera(self):
        if self.mode == "live" and self.camera and self.camera.is_opened:
            frame = self.camera.get_frame()
            if frame is not None:
                frame = cv2.resize(frame, (self.screen_width, self.screen_height))
                new_platforms = self.platform_detector.detect_platforms_from_frame(
                    frame, debug=False
                )
                if new_platforms:
                    self.platforms = new_platforms

    def reload_image_platforms(self):
        print("Перезагрузка площадок из doska.jpg...")
        new_platforms = self.platform_detector.detect_platforms_from_image(
            "doska.jpg", debug=self.debug
        )
        if new_platforms:
            self.platforms = new_platforms
            print(f"Загружено {len(new_platforms)} площадок")
        else:
            print("Площадки не найдены")

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed()

        current_size = self.screen.get_size()
        scale_x = self.screen_width / current_size[0]
        scale_y = self.screen_height / current_size[1]

        game_mouse_x = int(mouse_pos[0] * scale_x)
        game_mouse_y = int(mouse_pos[1] * scale_y)
        game_mouse_pos = (game_mouse_x, game_mouse_y)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_RETURN and (
                    keys[pygame.K_LALT] or keys[pygame.K_RALT]
                ):
                    self.toggle_fullscreen()
                elif event.key == pygame.K_ESCAPE:

                    if self.fullscreen:
                        self.toggle_fullscreen()
                    else:
                        self.running = False
                elif event.key == pygame.K_SPACE:
                    self.ball = Ball(100, 50)
                elif event.key == pygame.K_LEFT:
                    self.ball.vx -= 3
                elif event.key == pygame.K_RIGHT:
                    self.ball.vx += 3
                elif event.key == pygame.K_UP:
                    self.ball.vy -= 5
                elif event.key == pygame.K_DOWN:
                    self.ball.vy += 3
                elif event.key == pygame.K_r:
                    if self.mode == "live":
                        self.update_platforms_from_camera()
                    else:
                        self.reload_image_platforms()
                elif event.key == pygame.K_c and self.mode == "test":
                    self.platforms = []
                    if self.drawing_platform:
                        self.drawing_platform.completed_contours = []
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.mode == "test" and self.drawing_platform:
                        self.drawing_platform.start_drawing(*game_mouse_pos)
                elif event.button == 3:
                    self.ball.x, self.ball.y = game_mouse_pos
                    self.ball.vx = self.ball.vy = 0

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.mode == "test" and self.drawing_platform:
                    new_platform = self.drawing_platform.finish_drawing()
                    if new_platform:
                        self.platforms.append(new_platform)

            elif event.type == pygame.MOUSEMOTION:
                if (
                    self.mode == "test"
                    and self.drawing_platform
                    and self.drawing_platform.drawing
                ):
                    self.drawing_platform.add_point(*game_mouse_pos)

    def draw(self):

        self.screen.fill(self.BLACK)

        current_size = self.screen.get_size()
        scale_x = current_size[0] / self.screen_width
        scale_y = current_size[1] / self.screen_height

        for i, platform in enumerate(self.platforms):

            color = platform.color

            if len(platform.simplified_contour) > 2:
                points = platform.simplified_contour.reshape(-1, 2)

                scaled_points = [
                    (int(x * scale_x), int(y * scale_y)) for x, y in points
                ]
                pygame.draw.polygon(self.screen, color, scaled_points)
                pygame.draw.polygon(self.screen, self.WHITE, scaled_points, 2)

            if self.debug and self.mode == "test":
                font_small = pygame.font.Font(None, 20)
                num_text = font_small.render(str(i), True, self.RED)
                scaled_x = int(platform.x * scale_x + 5)
                scaled_y = int(platform.y * scale_y + 5)
                self.screen.blit(num_text, (scaled_x, scaled_y))

                for (x1, y1), (x2, y2) in platform.segments:
                    scaled_start = (int(x1 * scale_x), int(y1 * scale_y))
                    scaled_end = (int(x2 * scale_x), int(y2 * scale_y))
                    pygame.draw.line(self.screen, self.RED, scaled_start, scaled_end, 1)

        if (
            self.mode == "test"
            and self.drawing_platform
            and self.drawing_platform.drawing
            and len(self.drawing_platform.current_points) > 1
        ):
            points = self.drawing_platform.current_points

            scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in points]
            pygame.draw.lines(self.screen, self.ORANGE, False, scaled_points, 3)

        if len(self.ball.trail) > 1:
            for i in range(1, len(self.ball.trail)):
                alpha = i / len(self.ball.trail)
                color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))

                start_pos = (
                    int(self.ball.trail[i - 1][0] * scale_x),
                    int(self.ball.trail[i - 1][1] * scale_y),
                )
                end_pos = (
                    int(self.ball.trail[i][0] * scale_x),
                    int(self.ball.trail[i][1] * scale_y),
                )

                if i < len(self.ball.trail) - 1:
                    pygame.draw.line(
                        self.screen, color, start_pos, end_pos, max(1, int(3 * alpha))
                    )

        ball_color = self.WHITE
        scaled_ball_x = int(self.ball.x * scale_x)
        scaled_ball_y = int(self.ball.y * scale_y)
        scaled_radius = int(self.ball.radius * min(scale_x, scale_y))

        pygame.draw.circle(
            self.screen,
            ball_color,
            (scaled_ball_x, scaled_ball_y),
            scaled_radius,
        )

        if self.mode == "test":
            font_small = pygame.font.Font(None, 20)

            info_lines = [
                f"Режим: {self.mode.upper()} | Площадок: {len(self.platforms)} | Debug: {'ON' if self.debug else 'OFF'} | {'FULLSCREEN' if self.fullscreen else 'WINDOWED'}",
                "ЛКМ - рисовать платформу | ПКМ - переместить шарик",
                "SPACE - сброс | Стрелки - толчки | C - очистить | R - перезагрузить",
                "Alt+Enter - переключить полноэкранный режим | ESC - выход/переключить в окно",
            ]

            for i, line in enumerate(info_lines):
                text = font_small.render(line, True, self.WHITE)
                self.screen.blit(text, (10, 10 + i * 22))

            speed_text = font_small.render(
                f"Скорость: vx={self.ball.vx:.1f}, vy={self.ball.vy:.1f}",
                True,
                self.WHITE,
            )
            self.screen.blit(speed_text, (10, 100))

        pygame.display.flip()

    def run(self):
        print(f"Запуск векторной симуляции в режиме: {self.mode}")
        print(f"Разрешение: {self.screen_width}x{self.screen_height}")
        print(f"Режим экрана: {'полноэкранный' if self.fullscreen else 'оконный'}")
        if self.debug:
            print("DEBUG режим включен - будет показана обработка изображения")
        print("Управление:")
        if self.mode == "test":
            print("- ЛКМ: рисовать платформу")
            print("- ПКМ: переместить шарик")
            print("- C: очистить платформы")
        print("- SPACE: сброс шарика")
        print("- Стрелки: толчки")
        print("- R: перезагрузить платформы")
        print("- Alt+Enter: переключить полноэкранный режим")
        print("- ESC: выход или переключение в оконный режим")

        while self.running:
            current_time = time.time()

            self.handle_events()

            if (
                self.mode == "live"
                and current_time - self.last_platform_update
                > self.platform_update_interval
            ):
                self.update_platforms_from_camera()
                self.last_platform_update = current_time

            self.ball.update(self.platforms, self.screen_width, self.screen_height)

            self.draw()

            self.clock.tick(60)

        if self.camera:
            self.camera.release()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Симуляция падения шарика")
    parser.add_argument(
        "--mode",
        choices=["test", "live"],
        default="test",
        help="Режим: test (редактирование) или live (проектор)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="ID камеры (по умолчанию 0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Включить debug режим с показом этапов обработки изображения",
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="Ширина экрана (по умолчанию 1920)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="Высота экрана (по умолчанию 1080)"
    )
    parser.add_argument(
        "--windowed", action="store_true", help="Запустить в оконном режиме"
    )

    args = parser.parse_args()

    simulation = Game(
        mode=args.mode,
        camera_id=args.camera,
        debug=args.debug,
        screen_width=args.width,
        screen_height=args.height,
    )

    if args.windowed:
        simulation.toggle_fullscreen()

    simulation.run()


if __name__ == "__main__":
    main()
