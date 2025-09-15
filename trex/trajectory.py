#!/usr/bin/env python3
import numpy as np
from typing import Dict, List, Tuple, Optional
from logger import Logger


class TrajectoryCalculator:
    def __init__(self, logger: Logger):
        self.logger = logger

        # T-Rex физические константы
        self.GRAVITY = 0.6
        self.JUMP_VELOCITY = -10.0
        self.TREX_WIDTH = 44
        self.TREX_HEIGHT = 47
        self.GROUND_Y = 93

        # Константы скорости игры
        self.MIN_SPEED = 6.0
        self.MAX_SPEED = 13.0
        self.SPEED_ACCELERATION = 0.001

        # Margins безопасности
        self.COLLISION_MARGIN = 3
        self.JUMP_DISTANCE_MULTIPLIER = 1.1

        # Константы для птеродактилей
        self.PTERODACTYL_HEIGHTS = {
            "high": (20, 50),  # Высокие - всегда прыгать
            "medium": (50, 80),  # Средние - можно пригнуться или прыгнуть
            "low": (80, 110),  # Низкие - только пригнуться
        }

    def get_current_speed(self, game_time: float) -> float:
        speed = self.MIN_SPEED + (game_time * self.SPEED_ACCELERATION)
        return min(speed, self.MAX_SPEED)

    def should_jump_or_duck(
        self, trex_pos: Dict, obstacles: List[Dict], game_time: float
    ) -> Tuple[str, str]:
        try:
            # Проверяем состояние T-Rex
            if trex_pos["y"] < self.GROUND_Y - 5:
                return "none", "Already jumping"

            current_speed = self.get_current_speed(game_time)
            trex_x = trex_pos["center_x"]

            # Анализируем препятствия в зоне действия
            action_distance = current_speed * 100  # Смотрим на 100 кадров вперед
            threats = []

            for obstacle in obstacles:
                if not obstacle.get("is_threat", False):
                    continue

                obstacle_x = obstacle["center_x"]
                distance = obstacle_x - trex_x

                if 0 < distance < action_distance:
                    threat_info = self._analyze_threat(
                        obstacle, distance, current_speed, game_time
                    )
                    if threat_info:
                        threats.append(threat_info)

            if not threats:
                return "none", "No threats detected"

            # Сортируем угрозы по близости
            threats.sort(key=lambda t: t["distance"])
            nearest_threat = threats[0]

            # Принимаем решение на основе типа угрозы
            return self._decide_action(
                nearest_threat, trex_pos, current_speed, game_time
            )

        except Exception as e:
            self.logger.error("Error in action decision", e)
            return "none", f"Error: {str(e)}"

    def _analyze_threat(
        self, obstacle: Dict, distance: float, speed: float, game_time: float
    ) -> Optional[Dict]:
        try:
            # Время до столкновения
            time_to_collision = distance / speed

            threat_info = {
                "obstacle": obstacle,
                "distance": distance,
                "time_to_collision": time_to_collision,
                "type": obstacle["type"],
                "action_required": "none",
            }

            # Анализ птеродактилей
            if "pterodactyl" in obstacle["type"]:
                y_pos = obstacle["center_y"]

                if y_pos < 50:  # Высокий
                    threat_info["pterodactyl_level"] = "high"
                    threat_info["action_required"] = "jump"
                elif y_pos < 80:  # Средний
                    threat_info["pterodactyl_level"] = "medium"
                    # Для средних птеродактилей выбираем пригибание как более безопасное
                    threat_info["action_required"] = "duck"
                else:  # Низкий
                    threat_info["pterodactyl_level"] = "low"
                    threat_info["action_required"] = "duck"

            # Анализ кактусов
            elif "cactus" in obstacle["type"]:
                threat_info["action_required"] = "jump"

            # Проверяем временные рамки для действия
            if time_to_collision < 10:  # Очень близко
                threat_info["urgency"] = "critical"
            elif time_to_collision < 30:  # Близко
                threat_info["urgency"] = "high"
            elif time_to_collision < 60:  # Умеренно
                threat_info["urgency"] = "medium"
            else:  # Далеко
                threat_info["urgency"] = "low"

            return threat_info

        except Exception as e:
            self.logger.error("Error analyzing threat", e)
            return None

    def _decide_action(
        self, threat: Dict, trex_pos: Dict, speed: float, game_time: float
    ) -> Tuple[str, str]:
        try:
            obstacle = threat["obstacle"]
            required_action = threat["action_required"]
            urgency = threat["urgency"]
            distance = threat["distance"]

            # Критическая близость - немедленное действие
            if urgency == "critical":
                if required_action == "jump":
                    return "jump", f"Critical jump over {obstacle['type']}"
                elif required_action == "duck":
                    return "duck", f"Critical duck under {obstacle['type']}"

            # Оптимальное время для действия
            optimal_action_distance = speed * 45  # ~45 кадров до препятствия

            if required_action == "jump":
                # Проверяем безопасность прыжка
                if self._is_jump_safe(trex_pos, obstacle, speed, game_time):
                    if distance <= optimal_action_distance * 1.2:
                        return "jump", f"Safe jump over {obstacle['type']}"
                else:
                    return "none", f"Jump over {obstacle['type']} not safe"

            elif required_action == "duck":
                # Пригибание безопаснее для птеродактилей
                if distance <= optimal_action_distance:
                    return "duck", f"Duck under {obstacle['type']}"

            return "none", f"Waiting for optimal timing for {obstacle['type']}"

        except Exception as e:
            self.logger.error("Error deciding action", e)
            return "none", f"Decision error: {str(e)}"

    def _is_jump_safe(
        self, trex_pos: Dict, obstacle: Dict, speed: float, game_time: float
    ) -> bool:
        try:
            # Рассчитываем траекторию прыжка
            trajectory = self.calculate_jump_trajectory(
                trex_pos["center_x"], trex_pos["bottom_y"], game_time
            )

            if not trajectory:
                return False

            # Проверяем столкновения по траектории
            return not self._trajectory_collides_with_obstacle(
                trajectory, obstacle, speed
            )

        except Exception as e:
            self.logger.error("Error checking jump safety", e)
            return False

    def calculate_jump_trajectory(
        self, start_x: float, start_y: float, game_time: float
    ) -> List[Tuple[float, float]]:
        try:
            trajectory = []
            current_speed = self.get_current_speed(game_time)

            x = start_x
            y = start_y
            velocity_y = self.JUMP_VELOCITY
            time_step = 1.0

            # Рассчитываем траекторию до приземления
            while y <= self.GROUND_Y and len(trajectory) < 150:
                trajectory.append((x, y))

                x += current_speed * time_step
                y += velocity_y * time_step
                velocity_y += self.GRAVITY * time_step

            # Добавляем точку приземления
            if trajectory:
                trajectory.append((x, self.GROUND_Y))

            return trajectory

        except Exception as e:
            self.logger.error("Error calculating jump trajectory", e)
            return []

    def _trajectory_collides_with_obstacle(
        self, trajectory: List[Tuple[float, float]], obstacle: Dict, speed: float
    ) -> bool:
        try:
            obs_x = obstacle["x"] - self.COLLISION_MARGIN
            obs_y = obstacle["y"] - self.COLLISION_MARGIN
            obs_width = obstacle["width"] + (2 * self.COLLISION_MARGIN)
            obs_height = obstacle["height"] + (2 * self.COLLISION_MARGIN)

            for i, (trex_x, trex_y) in enumerate(trajectory):
                # Позиция препятствия во времени
                time_offset = i * 1.0
                obstacle_x_at_time = obs_x - (speed * time_offset)

                # Проверка пересечения хитбоксов
                if self._rectangles_intersect(
                    trex_x - self.TREX_WIDTH // 2,
                    trex_y - self.TREX_HEIGHT,
                    self.TREX_WIDTH,
                    self.TREX_HEIGHT,
                    obstacle_x_at_time,
                    obs_y,
                    obs_width,
                    obs_height,
                ):
                    return True

            return False

        except Exception as e:
            self.logger.error("Error checking trajectory collision", e)
            return True

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

    def calculate_optimal_action_timing(
        self, obstacle: Dict, trex_pos: Dict, game_time: float
    ) -> float:
        try:
            current_speed = self.get_current_speed(game_time)
            distance = obstacle["center_x"] - trex_pos["center_x"]

            if "pterodactyl" in obstacle["type"]:
                # Для птеродактилей оптимальное время пригибания
                return max(0, (distance - current_speed * 20) / current_speed)
            else:
                # Для кактусов оптимальное время прыжка
                jump_duration = 50  # Примерная длительность прыжка
                optimal_distance = current_speed * jump_duration * 0.6
                return max(0, (distance - optimal_distance) / current_speed)

        except Exception as e:
            self.logger.error("Error calculating optimal timing", e)
            return 0.0

    def analyze_multiple_threats(
        self, threats: List[Dict], trex_pos: Dict, game_time: float
    ) -> Tuple[str, str]:
        try:
            if not threats:
                return "none", "No threats"

            current_speed = self.get_current_speed(game_time)

            # Группируем угрозы по времени столкновения
            immediate_threats = [t for t in threats if t["time_to_collision"] < 30]
            upcoming_threats = [t for t in threats if 30 <= t["time_to_collision"] < 80]

            if immediate_threats:
                # Обрабатываем ближайшую угрозу
                nearest = min(immediate_threats, key=lambda t: t["time_to_collision"])
                return self._decide_action(nearest, trex_pos, current_speed, game_time)

            if upcoming_threats:
                # Готовимся к предстоящим угрозам
                next_threat = min(
                    upcoming_threats, key=lambda t: t["time_to_collision"]
                )

                # Проверяем, не создаст ли действие конфликт с другими угрозами
                if self._action_creates_conflict(
                    next_threat, upcoming_threats, current_speed
                ):
                    return "none", "Waiting to avoid conflict"

                return self._decide_action(
                    next_threat, trex_pos, current_speed, game_time
                )

            return "none", "All threats are distant"

        except Exception as e:
            self.logger.error("Error analyzing multiple threats", e)
            return "none", f"Analysis error: {str(e)}"

    def _action_creates_conflict(
        self, primary_threat: Dict, all_threats: List[Dict], speed: float
    ) -> bool:
        try:
            action = primary_threat["action_required"]
            action_duration = 50 if action == "jump" else 20  # duck shorter than jump

            primary_collision_time = primary_threat["time_to_collision"]
            action_end_time = primary_collision_time + action_duration

            # Проверяем, попадут ли другие угрозы в период действия
            for threat in all_threats:
                if threat == primary_threat:
                    continue

                other_collision_time = threat["time_to_collision"]

                if primary_collision_time <= other_collision_time <= action_end_time:
                    # Конфликт: другая угроза появится во время выполнения действия
                    return True

            return False

        except Exception as e:
            self.logger.error("Error checking action conflicts", e)
            return True  # В случае ошибки лучше быть осторожным
