import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from logger import Logger


class ObjectDetector:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.templates: Dict[str, np.ndarray] = {}
        self.confidence_threshold = 0.6
        self.low_confidence_threshold = 0.4
        self.load_templates()

    def load_templates(self) -> None:
        try:
            templates_path = "images"
            if not os.path.exists(templates_path):
                self.logger.error(f"Templates directory {templates_path} not found!")
                return

            template_files = {
                "trex": "trex.png",
                "dead_trex": "dead_trex.png",
                "trex_jumping": "trex_in_jump.png",
                "cactus_small": "mini_cactus.png",
                "cactus_large": "cactus.png",
                "cactus_double": "double_cactus.png",
                "cactus_triple": "triple_cactus.png",
                "cactus_double_small": "double_mini_cactus.png",
                "cactus_triple_small": "triple_mini_cactus.png",
                "pterodactyl": "pterodactyl.png",
                "game_over": "gameover.png",
            }

            loaded_count = 0
            for obj_type, filename in template_files.items():
                filepath = os.path.join(templates_path, filename)
                if os.path.exists(filepath):
                    template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.templates[obj_type] = template
                        loaded_count += 1
                        self.logger.info(
                            f"Loaded template {obj_type}: {template.shape}"
                        )
                    else:
                        self.logger.warning(f"Failed to load image: {filepath}")
                else:
                    self.logger.warning(f"Template file not found: {filepath}")

            png_files = [f for f in os.listdir(templates_path) if f.endswith(".png")]
            known_files = list(template_files.values()) + ["canvas.png"]

            for png_file in png_files:
                if png_file not in known_files:
                    obj_type = png_file.replace(".png", "")
                    filepath = os.path.join(templates_path, png_file)
                    template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.templates[obj_type] = template
                        loaded_count += 1
                        self.logger.info(
                            f"Loaded additional template {obj_type}: {template.shape}"
                        )

            self.logger.info(f"Loaded {loaded_count} templates from {templates_path}")

        except Exception as e:
            self.logger.error("Error loading templates", e)

    def detect_objects(self, screenshot: np.ndarray) -> List[Dict]:
        detected_objects = []

        try:
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # Улучшенная предобработка изображения
            gray_screenshot = cv2.GaussianBlur(gray_screenshot, (3, 3), 0)

            obstacle_types = [
                "cactus_small",
                "cactus_large",
                "cactus_double",
                "cactus_triple",
                "cactus_double_small",
                "cactus_triple_small",
                "pterodactyl",
            ]

            for obj_type in obstacle_types:
                if obj_type in self.templates:
                    objects = self._detect_template_objects_enhanced(
                        gray_screenshot, obj_type, self.templates[obj_type]
                    )
                    detected_objects.extend(objects)

            detected_objects = self._filter_overlapping_objects(detected_objects)
            detected_objects = self._classify_pterodactyl_heights(detected_objects)

            return detected_objects

        except Exception as e:
            self.logger.error("Error detecting objects", e)
            return []

    def detect_trex(self, screenshot: np.ndarray) -> Optional[Dict]:
        try:
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            gray_screenshot = cv2.GaussianBlur(gray_screenshot, (3, 3), 0)

            trex_states = [
                ("dead", "dead_trex"),
                ("jumping", "trex_jumping"),
                ("running", "trex"),
            ]

            best_match = None
            best_confidence = 0

            for state, template_key in trex_states:
                if template_key not in self.templates:
                    continue

                template = self.templates[template_key]

                # Мульти-масштабное сопоставление
                for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    resized_template = cv2.resize(template, None, fx=scale, fy=scale)

                    if (
                        resized_template.shape[0] > gray_screenshot.shape[0]
                        or resized_template.shape[1] > gray_screenshot.shape[1]
                    ):
                        continue

                    result = cv2.matchTemplate(
                        gray_screenshot, resized_template, cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if (
                        max_val > self.low_confidence_threshold
                        and max_val > best_confidence
                    ):
                        template_h, template_w = resized_template.shape
                        x, y = max_loc

                        best_match = {
                            "state": state,
                            "x": x,
                            "y": y,
                            "width": template_w,
                            "height": template_h,
                            "center_x": x + template_w // 2,
                            "center_y": y + template_h // 2,
                            "bottom_y": y
                            + template_h
                            + (5 if template_key == "trex" else 0),
                            "confidence": float(max_val),
                            "scale": scale,
                        }
                        best_confidence = max_val

            return best_match

        except Exception as e:
            self.logger.error("Error detecting T-Rex", e)
            return None

    def detect_game_over(self, screenshot: np.ndarray) -> bool:
        try:
            if "game_over" not in self.templates:
                return False

            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            template = self.templates["game_over"]

            # Проверяем разные части экрана для game over
            regions = [
                gray_screenshot,  # Весь экран
                gray_screenshot[:100, :],  # Верхняя часть
                gray_screenshot[25:125, :],  # Средняя часть
            ]

            for region in regions:
                if (
                    region.shape[0] < template.shape[0]
                    or region.shape[1] < template.shape[1]
                ):
                    continue

                result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > 0.7:
                    return True

            return False

        except Exception as e:
            self.logger.error("Error detecting game over", e)
            return False

    def _detect_template_objects_enhanced(
        self, gray_image: np.ndarray, obj_type: str, template: np.ndarray
    ) -> List[Dict]:
        objects = []

        try:
            img_h, img_w = gray_image.shape
            template_h, template_w = template.shape

            if template_h > img_h or template_w > img_w:
                return objects

            # Используем разные пороги для разных типов объектов
            threshold = self.confidence_threshold
            if "pterodactyl" in obj_type:
                threshold = 0.5  # Птеродактили сложнее обнаружить
            elif "cactus" in obj_type:
                threshold = 0.65  # Кактусы проще

            # Мульти-масштабное обнаружение
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                resized_template = cv2.resize(template, None, fx=scale, fy=scale)

                if (
                    resized_template.shape[0] > img_h
                    or resized_template.shape[1] > img_w
                ):
                    continue

                result = cv2.matchTemplate(
                    gray_image, resized_template, cv2.TM_CCOEFF_NORMED
                )
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):
                    x, y = pt
                    confidence = result[y, x]

                    scaled_w = int(template_w * scale)
                    scaled_h = int(template_h * scale)

                    obj_data = {
                        "type": obj_type,
                        "x": x,
                        "y": y,
                        "width": scaled_w,
                        "height": scaled_h,
                        "center_x": x + scaled_w // 2,
                        "center_y": y + scaled_h // 2,
                        "confidence": float(confidence),
                        "is_threat": True,
                        "scale": scale,
                    }

                    if "pterodactyl" in obj_type:
                        obj_data["is_flying"] = True

                    objects.append(obj_data)

        except Exception as e:
            self.logger.error(f"Error detecting template {obj_type}", e)

        return objects

    def _classify_pterodactyl_heights(self, objects: List[Dict]) -> List[Dict]:
        for obj in objects:
            if "pterodactyl" in obj["type"]:
                y_pos = obj["center_y"]

                # Классификация высоты птеродактиля
                if y_pos < 40:
                    obj["height_level"] = "high"
                    obj["duck_required"] = False
                elif y_pos < 70:
                    obj["height_level"] = "medium"
                    obj["duck_required"] = True  # Можно пригнуться или прыгнуть
                else:
                    obj["height_level"] = "low"
                    obj["duck_required"] = True

        return objects

    def _filter_overlapping_objects(self, objects: List[Dict]) -> List[Dict]:
        if not objects:
            return objects

        sorted_objects = sorted(objects, key=lambda x: x["confidence"], reverse=True)
        filtered_objects = []

        for obj in sorted_objects:
            is_overlapping = False

            for filtered_obj in filtered_objects:
                if self._calculate_overlap(obj, filtered_obj) > 0.2:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered_objects.append(obj)

        return filtered_objects

    def _calculate_overlap(self, obj1: Dict, obj2: Dict) -> float:
        try:
            x1 = max(obj1["x"], obj2["x"])
            y1 = max(obj1["y"], obj2["y"])
            x2 = min(obj1["x"] + obj1["width"], obj2["x"] + obj2["width"])
            y2 = min(obj1["y"] + obj1["height"], obj2["y"] + obj2["height"])

            if x2 <= x1 or y2 <= y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)
            area1 = obj1["width"] * obj1["height"]
            area2 = obj2["width"] * obj2["height"]
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = max(0.1, min(0.99, threshold))
        self.logger.info(f"Confidence threshold set to: {self.confidence_threshold}")

    def get_template_info(self) -> Dict[str, Dict]:
        info = {}
        for obj_type, template in self.templates.items():
            info[obj_type] = {
                "shape": template.shape,
                "is_threat": obj_type
                not in {"trex", "dead_trex", "trex_jumping", "game_over"},
                "loaded": True,
            }
        return info
