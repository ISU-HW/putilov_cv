import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from debug import Logger


class TrajectoryCalculator:
    def __init__(self, logger: Logger):
        self.logger = logger

        self.GRAVITY = 0.6
        self.JUMP_VELOCITY = -10.0
        self.TREX_HEIGHT = 47
        self.TREX_WIDTH = 44
        self.GROUND_Y = 93

        self.MIN_SPEED = 6.0
        self.MAX_SPEED = 13.0
        self.ACCELERATION = 0.001

        self.trajectory_cache: Dict[str, List[Tuple[float, float]]] = {}

    def get_current_speed(self, game_time: float) -> float:
        speed = self.MIN_SPEED + (game_time * self.ACCELERATION)
        return min(speed, self.MAX_SPEED)

    def calculate_jump_trajectory(
        self, start_x: float, start_y: float, game_time: float
    ) -> List[Tuple[float, float]]:
        start_time = time.time()

        try:

            cache_key = f"{start_x}_{start_y}_{game_time:.2f}"
            if cache_key in self.trajectory_cache:
                return self.trajectory_cache[cache_key]

            trajectory = []
            current_speed = self.get_current_speed(game_time)

            x = start_x
            y = start_y
            velocity_y = self.JUMP_VELOCITY
            time_step = 1.0

            while y <= self.GROUND_Y:
                trajectory.append((x, y))

                x += current_speed * time_step
                y += velocity_y * time_step
                velocity_y += self.GRAVITY * time_step

                if len(trajectory) > 1000:
                    self.logger.warning("Траектория прыжка превысила лимит точек")
                    break

            trajectory.append((x, self.GROUND_Y))

            if len(self.trajectory_cache) < 100:
                self.trajectory_cache[cache_key] = trajectory
            elif len(self.trajectory_cache) >= 100:

                self.trajectory_cache.clear()
                self.trajectory_cache[cache_key] = trajectory

            processing_time = (time.time() - start_time) * 1000
            if processing_time > 5:
                self.logger.warning(f"Расчет траектории занял {processing_time:.1f}мс")

            return trajectory

        except Exception as e:
            self.logger.error("Ошибка расчета траектории прыжка", e)
            return []

    def check_collision_with_objects(
        self,
        trajectory: List[Tuple[float, float]],
        objects: List[Dict],
        game_time: float,
    ) -> Tuple[bool, Optional[Dict]]:
        start_time = time.time()

        try:
            if not trajectory or not objects:
                return False, None

            current_speed = self.get_current_speed(game_time)

            threats = [obj for obj in objects if obj.get("is_threat", False)]

            for threat in threats:
                bbox = threat["bbox"]

                for i, (trex_x, trex_y) in enumerate(trajectory):
                    time_offset = i * 1.0

                    obstacle_x = bbox["x"] - (current_speed * time_offset)
                    obstacle_y = bbox["y"]
                    obstacle_width = bbox["width"]
                    obstacle_height = bbox["height"]

                    if self._rectangles_intersect(
                        trex_x,
                        trex_y,
                        self.TREX_WIDTH,
                        self.TREX_HEIGHT,
                        obstacle_x,
                        obstacle_y,
                        obstacle_width,
                        obstacle_height,
                    ):
                        processing_time = (time.time() - start_time) * 1000
                        if processing_time > 3:
                            self.logger.warning(
                                f"Проверка коллизий заняла {processing_time:.1f}мс"
                            )

                        return True, threat

            processing_time = (time.time() - start_time) * 1000
            if processing_time > 3:
                self.logger.warning(f"Проверка коллизий заняла {processing_time:.1f}мс")

            return False, None

        except Exception as e:
            self.logger.error("Ошибка проверки коллизий", e)
            return (
                True,
                None,
            )

    def _rectangles_intersect(
        self,
        x1: float,
        y1: float,
        w1: float,
        h1: float,
        x2: float,
        y2: float,
        w2: float,
        h2: float,
    ) -> bool:
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def should_jump(
        self, trex_position: Tuple[float, float], objects: List[Dict], game_time: float
    ) -> Tuple[bool, str]:
        start_time = time.time()

        try:
            trex_x, trex_y = trex_position

            if trex_y < self.GROUND_Y - 5:
                return False, "T-Rex уже в прыжке"

            current_speed = self.get_current_speed(game_time)
            jump_distance = current_speed * 60

            nearby_threats = []
            for obj in objects:
                if obj.get("is_threat", False):
                    obj_x = obj["bbox"]["x"]

                    if trex_x < obj_x < trex_x + jump_distance * 1.5:
                        nearby_threats.append(obj)

            if not nearby_threats:
                return False, "Нет угроз в зоне прыжка"

            trajectory = self.calculate_jump_trajectory(trex_x, trex_y, game_time)

            if not trajectory:
                return False, "Не удалось рассчитать траекторию"

            will_collide, collision_obj = self.check_collision_with_objects(
                trajectory, nearby_threats, game_time
            )

            processing_time = (time.time() - start_time) * 1000
            if processing_time > 10:
                self.logger.warning(
                    f"Принятие решения заняло {processing_time:.1f}мс (превышен лимит 10мс)"
                )

            if will_collide:
                collision_type = (
                    collision_obj["type"] if collision_obj else "неизвестное"
                )
                return False, f"Прыжок приведет к столкновению с {collision_type}"
            else:
                threat_types = [obj["type"] for obj in nearby_threats]
                return True, f"Безопасный прыжок через {', '.join(threat_types)}"

        except Exception as e:
            self.logger.error("Ошибка принятия решения о прыжке", e)
            return False, f"Ошибка расчета: {str(e)}"
