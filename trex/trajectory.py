import numpy as np
from typing import Dict, List, Tuple, Optional
from logger import Logger


class TrajectoryCalculator:
    def __init__(self, logger: Logger):
        self.logger = logger

        self.GRAVITY = 0.6
        self.JUMP_VELOCITY = -10.0
        self.TREX_WIDTH = 44
        self.TREX_HEIGHT = 47
        self.GROUND_Y = 93

        self.TREX_HITBOX_BOTTOM_PADDING = 5

        self.MIN_SPEED = 6.0
        self.MAX_SPEED = 13.0
        self.SPEED_ACCELERATION = 0.001

        self.COLLISION_MARGIN = 5
        self.JUMP_DISTANCE_MULTIPLIER = 1.5

        self.PTERODACTYL_HEIGHTS = {
            "high": (20, 50),
            "medium": (50, 80),
            "low": (80, 110),
        }

    def get_current_speed(self, game_time: float) -> float:
        speed = self.MIN_SPEED + (game_time * self.SPEED_ACCELERATION)
        return min(speed, self.MAX_SPEED)

    def should_jump_or_duck(
        self, trex_pos: Dict, obstacles: List[Dict], game_time: float
    ) -> Tuple[str, str]:
        try:

            if trex_pos["y"] < self.GROUND_Y - 5:
                return "none", "Already jumping"

            current_speed = self.get_current_speed(game_time)
            trex_x = trex_pos["center_x"]

            action_distance = current_speed * 150
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

            threats.sort(key=lambda t: t["distance"])
            nearest_threat = threats[0]

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

            time_to_collision = distance / speed

            threat_info = {
                "obstacle": obstacle,
                "distance": distance,
                "time_to_collision": time_to_collision,
                "type": obstacle["type"],
                "action_required": "none",
            }

            if "pterodactyl" in obstacle["type"]:
                y_pos = obstacle["center_y"]

                if y_pos < 50:
                    threat_info["pterodactyl_level"] = "high"
                    threat_info["action_required"] = "jump"
                elif y_pos < 80:
                    threat_info["pterodactyl_level"] = "medium"

                    threat_info["action_required"] = "duck"
                else:
                    threat_info["pterodactyl_level"] = "low"
                    threat_info["action_required"] = "duck"

            elif "cactus" in obstacle["type"]:
                threat_info["action_required"] = "jump"

            if time_to_collision < 20:
                threat_info["urgency"] = "critical"
            elif time_to_collision < 50:
                threat_info["urgency"] = "high"
            elif time_to_collision < 90:
                threat_info["urgency"] = "medium"
            else:
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

            self.logger.info(
                f"DECISION: {obstacle['type']} at distance {distance:.1f}, urgency: {urgency}, action: {required_action}"
            )

            if urgency == "critical":
                if required_action == "jump":
                    self.logger.info(
                        f"JUMP DECISION: Critical jump over {obstacle['type']} at distance {distance:.1f}"
                    )
                    return "jump", f"Critical jump over {obstacle['type']}"
                elif required_action == "duck":
                    self.logger.info(
                        f"DUCK DECISION: Critical duck under {obstacle['type']} at distance {distance:.1f}"
                    )
                    return "duck", f"Critical duck under {obstacle['type']}"

            optimal_action_distance = speed * 70

            if required_action == "jump":

                if self._is_jump_safe(trex_pos, obstacle, speed, game_time):
                    if distance <= optimal_action_distance * 1.5:
                        self.logger.info(
                            f"JUMP DECISION: Safe jump over {obstacle['type']} at distance {distance:.1f}"
                        )
                        return "jump", f"Safe jump over {obstacle['type']}"
                else:
                    self.logger.info(
                        f"JUMP REJECTED: Jump over {obstacle['type']} not safe at distance {distance:.1f}"
                    )
                    return "none", f"Jump over {obstacle['type']} not safe"

            elif required_action == "duck":

                if distance <= optimal_action_distance * 1.2:
                    self.logger.info(
                        f"DUCK DECISION: Duck under {obstacle['type']} at distance {distance:.1f}"
                    )
                    return "duck", f"Duck under {obstacle['type']}"

            return (
                "none",
                f"Waiting for optimal timing for {obstacle['type']} (distance: {distance:.1f})",
            )

        except Exception as e:
            self.logger.error("Error deciding action", e)
            return "none", f"Decision error: {str(e)}"

    def _is_jump_safe(
        self, trex_pos: Dict, obstacle: Dict, speed: float, game_time: float
    ) -> bool:
        try:

            trajectory = self.calculate_jump_trajectory(
                trex_pos["center_x"],
                trex_pos["bottom_y"] + self.TREX_HITBOX_BOTTOM_PADDING,
                game_time,
            )

            if not trajectory:
                return False

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

            while y <= self.GROUND_Y and len(trajectory) < 150:
                trajectory.append((x, y))

                x += current_speed * time_step
                y += velocity_y * time_step
                velocity_y += self.GRAVITY * time_step

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

                time_offset = i * 1.0
                obstacle_x_at_time = obs_x - (speed * time_offset)

                trex_height_with_padding = (
                    self.TREX_HEIGHT + self.TREX_HITBOX_BOTTOM_PADDING
                )

                if self._rectangles_intersect(
                    trex_x - self.TREX_WIDTH // 2,
                    trex_y - trex_height_with_padding,
                    self.TREX_WIDTH,
                    trex_height_with_padding,
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

                return max(0, (distance - current_speed * 30) / current_speed)
            else:

                jump_duration = 50
                optimal_distance = current_speed * jump_duration * 0.8
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

            immediate_threats = [t for t in threats if t["time_to_collision"] < 50]
            upcoming_threats = [
                t for t in threats if 50 <= t["time_to_collision"] < 120
            ]

            if immediate_threats:

                nearest = min(immediate_threats, key=lambda t: t["time_to_collision"])
                return self._decide_action(nearest, trex_pos, current_speed, game_time)

            if upcoming_threats:

                next_threat = min(
                    upcoming_threats, key=lambda t: t["time_to_collision"]
                )

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
            action_duration = 50 if action == "jump" else 20

            primary_collision_time = primary_threat["time_to_collision"]
            action_end_time = primary_collision_time + action_duration

            for threat in all_threats:
                if threat == primary_threat:
                    continue

                other_collision_time = threat["time_to_collision"]

                if primary_collision_time <= other_collision_time <= action_end_time:

                    return True

            return False

        except Exception as e:
            self.logger.error("Error checking action conflicts", e)
            return True
