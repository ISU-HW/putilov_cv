import cv2
import numpy as np
import pygame
import math
import time
import argparse
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist


class VectorPlatform:
    def __init__(self, contour: np.ndarray):
        self.contour = contour
        self.simplified_contour = self._simplify_contour(contour)
        self.bbox = cv2.boundingRect(contour)
        self.x, self.y, self.w, self.h = self.bbox
        self.segments = self._create_segments()

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
    def __init__(self, min_area: int = 500):
        self.min_area = min_area
        self.last_platforms = []
        self.background_model = None

    def _preprocess_image(
        self, image: np.ndarray, block_size: int = 11, c_value: int = 2
    ) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        adaptive_thresh = cv2.adaptiveThreshold(
            opened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_close)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        eroded = cv2.erode(cleaned, kernel_erode, iterations=1)
        dilated = cv2.dilate(eroded, kernel_erode, iterations=1)

        return dilated

    def _find_optimal_threshold(self, image: np.ndarray) -> Tuple[int, int, int]:

        block_sizes = [9, 11, 13, 15, 17, 19]
        c_values = [1, 2, 3, 4, 5]

        best_params = (11, 2)
        best_score = -1
        results = []

        for block_size in block_sizes:
            for c_value in c_values:
                try:

                    processed = self._preprocess_image(image, block_size, c_value)
                    edges = cv2.Canny(processed, 30, 100)
                    contours, _ = cv2.findContours(
                        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    valid_count = 0
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > self.min_area:
                            perimeter = cv2.arcLength(contour, True)
                            if perimeter > 0:
                                circularity = (
                                    4 * math.pi * area / (perimeter * perimeter)
                                )
                                if circularity < 0.8:
                                    valid_count += 1

                    total_contours = len(contours)
                    if valid_count > 0:

                        noise_penalty = max(0, total_contours - 20) * 0.1
                        score = valid_count - noise_penalty
                    else:
                        score = -1

                    results.append(
                        (block_size, c_value, total_contours, valid_count, score)
                    )

                    if score > best_score:
                        best_score = score
                        best_params = (block_size, c_value)

                except Exception:
                    continue

        return best_params[0], best_params[1], results

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

            print("Поиск оптимальных параметров threshold...")
            best_block_size, best_c_value, threshold_results = (
                self._find_optimal_threshold(frame)
            )
            print(
                f"Выбраны параметры: block_size={best_block_size}, c_value={best_c_value}"
            )

            processed = self._preprocess_image(frame, best_block_size, best_c_value)

            edges = cv2.Canny(processed, 30, 100)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if debug:
                self._show_debug_stages(
                    original,
                    processed,
                    edges,
                    contours,
                    threshold_results,
                    best_block_size,
                    best_c_value,
                )

            platforms = []
            for contour in contours:
                area = cv2.contourArea(contour)

                if area > self.min_area:

                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        if circularity < 0.8:
                            platforms.append(VectorPlatform(contour))

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
        edges: np.ndarray,
        contours: list,
        threshold_results: list,
        best_block_size: int,
        best_c_value: int,
    ):
        try:

            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            if len(original.shape) == 3:
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(original, cmap="gray")
            plt.title("Оригинальное изображение")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(processed, cmap="gray")
            plt.title(
                f"Предобработка\n(block_size={best_block_size}, c={best_c_value})"
            )
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(edges, cmap="gray")
            plt.title("Детекция краев (Canny)")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            contour_img = original.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            if len(contour_img.shape) == 3:
                plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(contour_img, cmap="gray")
            plt.title(f"Все контуры ({len(contours)})")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            filtered_img = original.copy()
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        if circularity < 0.8:
                            valid_contours.append(contour)

            cv2.drawContours(filtered_img, valid_contours, -1, (255, 0, 0), 3)
            if len(filtered_img.shape) == 3:
                plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(filtered_img, cmap="gray")
            plt.title(f"Валидные площадки ({len(valid_contours)})")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.text(0.05, 0.9, f"Минимальная площадь: {self.min_area}", fontsize=11)
            plt.text(0.05, 0.8, f"Найдено контуров: {len(contours)}", fontsize=11)
            plt.text(
                0.05, 0.7, f"Валидных площадок: {len(valid_contours)}", fontsize=11
            )
            plt.text(0.05, 0.6, f"Лучшие параметры:", fontsize=11, weight="bold")
            plt.text(0.05, 0.5, f"block_size = {best_block_size}", fontsize=10)
            plt.text(0.05, 0.4, f"c_value = {best_c_value}", fontsize=10)
            plt.text(0.05, 0.3, "Критерии фильтрации:", fontsize=11, weight="bold")
            plt.text(0.05, 0.2, "- Площадь > min_area", fontsize=9)
            plt.text(0.05, 0.1, "- Круглость < 0.8", fontsize=9)
            plt.text(0.05, 0.0, "- Оптимизация по количеству", fontsize=9)
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            if threshold_results:
                self._show_threshold_analysis(
                    threshold_results, best_block_size, best_c_value
                )

        except Exception as e:
            print(f"Ошибка при показе debug информации: {e}")

    def _show_threshold_analysis(
        self, threshold_results: list, best_block_size: int, best_c_value: int
    ):
        """Показ анализа различных threshold параметров"""
        try:
            plt.figure(figsize=(12, 8))

            block_sizes = sorted(set([r[0] for r in threshold_results]))
            c_values = sorted(set([r[1] for r in threshold_results]))

            valid_matrix = np.zeros((len(c_values), len(block_sizes)))
            score_matrix = np.zeros((len(c_values), len(block_sizes)))

            for block_size, c_value, total, valid, score in threshold_results:
                i = c_values.index(c_value)
                j = block_sizes.index(block_size)
                valid_matrix[i, j] = valid
                score_matrix[i, j] = score if score > 0 else 0

            plt.subplot(2, 2, 1)
            im1 = plt.imshow(valid_matrix, cmap="viridis", aspect="auto")
            plt.colorbar(im1, label="Валидные контуры")
            plt.title("Количество валидных контуров")
            plt.xlabel("Block Size")
            plt.ylabel("C Value")
            plt.xticks(range(len(block_sizes)), block_sizes)
            plt.yticks(range(len(c_values)), c_values)

            best_i = c_values.index(best_c_value)
            best_j = block_sizes.index(best_block_size)
            plt.scatter(best_j, best_i, c="red", s=100, marker="x", linewidths=3)

            plt.subplot(2, 2, 2)
            im2 = plt.imshow(score_matrix, cmap="RdYlGn", aspect="auto")
            plt.colorbar(im2, label="Оценка качества")
            plt.title("Оценка качества параметров")
            plt.xlabel("Block Size")
            plt.ylabel("C Value")
            plt.xticks(range(len(block_sizes)), block_sizes)
            plt.yticks(range(len(c_values)), c_values)
            plt.scatter(best_j, best_i, c="blue", s=100, marker="x", linewidths=3)

            plt.subplot(2, 2, 3)
            best_results = sorted(threshold_results, key=lambda x: x[4], reverse=True)[
                :10
            ]
            table_data = []
            for i, (bs, cv, total, valid, score) in enumerate(best_results):
                marker = "★" if bs == best_block_size and cv == best_c_value else ""
                table_data.append([f"{marker} {bs}", cv, total, valid, f"{score:.2f}"])

            plt.table(
                cellText=table_data,
                colLabels=["Block Size", "C Value", "Всего", "Валидных", "Оценка"],
                cellLoc="center",
                loc="center",
            )
            plt.title("Топ-10 результатов")
            plt.axis("off")

            plt.subplot(2, 2, 4)
            total_tests = len(threshold_results)
            successful_tests = len([r for r in threshold_results if r[3] > 0])
            avg_valid = (
                np.mean([r[3] for r in threshold_results if r[3] > 0])
                if successful_tests > 0
                else 0
            )

            stats_text = f"""
            Всего тестов: {total_tests}
            Успешных: {successful_tests}
            Среднее количество валидных: {avg_valid:.1f}
            
            Выбранные параметры:
            Block Size: {best_block_size}
            C Value: {best_c_value}
            
            Критерий выбора:
            Максимальное количество валидных контуров
            с минимальным шумом
            """

            plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center")
            plt.title("Статистика оптимизации")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Ошибка при показе анализа threshold: {e}")


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
            platform = VectorPlatform(contour)
            self.completed_contours.append(contour)
            self.drawing = False
            self.current_points = []
            return platform

        self.drawing = False
        self.current_points = []
        return None

    def cancel_drawing(self):
        """Отменить текущее рисование"""
        self.drawing = False
        self.current_points = []


class CameraManager:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_opened = False

    def initialize(self) -> bool:
        """Инициализация камеры"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"Не удалось открыть камеру {self.camera_id}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

            self.is_opened = True
            print(f"Камера {self.camera_id} успешно инициализирована")
            return True

        except Exception as e:
            print(f"Ошибка инициализации камеры: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Получение кадра с камеры"""
        if not self.is_opened or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        """Освобождение ресурсов камеры"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False


class Game:
    def __init__(
        self,
        mode: str = "test",
        camera_id: int = 0,
        screen_width: int = 800,
        screen_height: int = 600,
        debug: bool = False,
    ):
        self.mode = mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.debug = debug

        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
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

    def update_platforms_from_camera(self):
        """Обновление площадок из кадра камеры"""
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
        """Перезагрузка площадок из изображения"""
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
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

                        self.drawing_platform.start_drawing(*mouse_pos)
                elif event.button == 3:

                    self.ball.x, self.ball.y = mouse_pos
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

                    self.drawing_platform.add_point(*mouse_pos)

    def draw(self):
        if self.mode == "live":
            self.screen.fill(self.BLACK)
        else:
            self.screen.fill(self.WHITE)

        for i, platform in enumerate(self.platforms):
            color = self.GREEN if self.mode == "live" else self.BLUE

            if len(platform.simplified_contour) > 2:
                points = platform.simplified_contour.reshape(-1, 2)
                pygame.draw.polygon(self.screen, color, points)

                pygame.draw.polygon(
                    self.screen,
                    self.BLACK if self.mode == "test" else self.WHITE,
                    points,
                    2,
                )

            if self.debug and self.mode == "test":
                font_small = pygame.font.Font(None, 20)
                num_text = font_small.render(str(i), True, self.RED)
                self.screen.blit(num_text, (platform.x + 5, platform.y + 5))

                for (x1, y1), (x2, y2) in platform.segments:
                    pygame.draw.line(self.screen, self.RED, (x1, y1), (x2, y2), 1)

        if (
            self.mode == "test"
            and self.drawing_platform
            and self.drawing_platform.drawing
            and len(self.drawing_platform.current_points) > 1
        ):
            points = self.drawing_platform.current_points
            pygame.draw.lines(self.screen, self.ORANGE, False, points, 3)

        if len(self.ball.trail) > 1:
            for i in range(1, len(self.ball.trail)):
                alpha = i / len(self.ball.trail)
                color = (int(255 * alpha), int(100 * alpha), int(100 * alpha))
                if self.mode == "live":
                    color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))

                start_pos = (
                    int(self.ball.trail[i - 1][0]),
                    int(self.ball.trail[i - 1][1]),
                )
                end_pos = (int(self.ball.trail[i][0]), int(self.ball.trail[i][1]))

                if i < len(self.ball.trail) - 1:
                    pygame.draw.line(
                        self.screen, color, start_pos, end_pos, max(1, int(3 * alpha))
                    )

        ball_color = self.WHITE if self.mode == "live" else self.RED
        pygame.draw.circle(
            self.screen,
            ball_color,
            (int(self.ball.x), int(self.ball.y)),
            int(self.ball.radius),
        )

        if self.mode == "test":
            shadow_y = self.ball.y + 8
            pygame.draw.ellipse(
                self.screen,
                (150, 150, 150),
                (self.ball.x - self.ball.radius, shadow_y - 2, self.ball.radius * 2, 4),
            )

        if self.mode == "test":
            font = pygame.font.Font(None, 28)
            font_small = pygame.font.Font(None, 20)

            info_lines = [
                f"Режим: {self.mode.upper()} | Площадок: {len(self.platforms)} | Debug: {'ON' if self.debug else 'OFF'}",
                "ЛКМ - рисовать площадку | ПКМ - переместить шарик",
                "SPACE - сброс | Стрелки - толчки | C - очистить | R - перезагрузить",
            ]

            for i, line in enumerate(info_lines):
                text = font_small.render(line, True, self.BLACK)
                self.screen.blit(text, (10, 10 + i * 22))

            speed_text = font_small.render(
                f"Скорость: vx={self.ball.vx:.1f}, vy={self.ball.vy:.1f}",
                True,
                self.BLACK,
            )
            self.screen.blit(speed_text, (10, 80))

        pygame.display.flip()

    def run(self):
        print(f"Запуск векторной симуляции в режиме: {self.mode}")
        if self.debug:
            print("DEBUG режим включен - будет показана обработка изображения")
        print("Управление:")
        if self.mode == "test":
            print("- ЛКМ: рисовать площадку")
            print("- ПКМ: переместить шарик")
            print("- C: очистить площадки")
        print("- SPACE: сброс шарика")
        print("- Стрелки: толчки")
        print("- R: перезагрузить площадки")

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
    parser = argparse.ArgumentParser(description="Векторная симуляция падения шарика")
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

    args = parser.parse_args()

    simulation = Game(mode=args.mode, camera_id=args.camera, debug=args.debug)
    simulation.run()


if __name__ == "__main__":
    main()
