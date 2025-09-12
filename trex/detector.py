import cv2
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
from debug import Logger


class ObjectDetector:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.templates: Dict[str, np.ndarray] = {}
        self.template_sizes: Dict[str, Tuple[int, int]] = {}
        self.confidence_threshold = 0.7
        self.load_templates()

    def load_templates(self) -> None:
        try:
            templates_path = "images"
            if not os.path.exists(templates_path):
                self.logger.error(f"Папка {templates_path} не найдена!")
                return

            template_files = {
                "trex": "trex.png",
                "dead_trex": "dead_trex.png",
                "trex_in_jump": "trex_in_jump.png",
                "cactus": "cactus.png",
                "mini_cactus": "mini_cactus.png",
                "triple_cactus": "triple_cactus.png",
                "triple_mini_cactus": "triple_mini_cactus.png",
                "double_cactus": "double_cactus.png",
                "double_mini_cactus": "double_mini_cactus.png",
                "pterodactyl": "pterodactyl.png",
            }

            loaded_count = 0
            for obj_type, filename in template_files.items():
                filepath = os.path.join(templates_path, filename)
                if os.path.exists(filepath):
                    template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.templates[obj_type] = template
                        self.template_sizes[obj_type] = template.shape[:2]
                        loaded_count += 1
                        self.logger.info(
                            f"Шаблон {obj_type} загружен: {template.shape}"
                        )
                    else:
                        self.logger.warning(
                            f"Не удалось загрузить изображение: {filepath}"
                        )
                else:
                    self.logger.warning(f"Файл шаблона не найден: {filepath}")

            png_files = [f for f in os.listdir(templates_path) if f.endswith(".png")]
            known_files = list(template_files.values()) + ["canvas.png", "gameover.png"]

            for png_file in png_files:
                if png_file not in known_files:
                    obj_type = png_file.replace(".png", "")
                    filepath = os.path.join(templates_path, png_file)
                    template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.templates[obj_type] = template
                        self.template_sizes[obj_type] = template.shape[:2]
                        loaded_count += 1
                        self.logger.info(
                            f"Дополнительный шаблон {obj_type} загружен: {template.shape}"
                        )

            self.logger.info(
                f"Загружено {loaded_count} шаблонов из папки {templates_path}"
            )

        except Exception as e:
            self.logger.error("Ошибка загрузки шаблонов", e)

    def detect_objects(self, screenshot: np.ndarray) -> List[Dict]:
        start_time = time.time()
        detected_objects = []

        try:
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            for obj_type, template in self.templates.items():
                objects = self._detect_template_objects(
                    gray_screenshot, obj_type, template
                )
                detected_objects.extend(objects)

            pterodactyls = self._detect_pterodactyl(gray_screenshot)
            detected_objects.extend(pterodactyls)

            detected_objects = self._filter_overlapping_objects(detected_objects)

            processing_time = (time.time() - start_time) * 1000
            if processing_time > 10:
                self.logger.warning(
                    f"Детекция заняла {processing_time:.1f}мс (превышен лимит 10мс)"
                )

            return detected_objects

        except Exception as e:
            self.logger.error("Ошибка детекции объектов", e)
            return []

    def detect_trex_state_and_position(self, screenshot: np.ndarray) -> Optional[Dict]:
        try:
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            trex_states = [
                ("dead", "dead_trex"),
                ("jumping", "trex_in_jump"),
                ("normal", "trex"),
            ]

            best_match = None
            best_confidence = 0

            for state, template_key in trex_states:
                if template_key not in self.templates:
                    continue

                template = self.templates[template_key]
                result = cv2.matchTemplate(
                    gray_screenshot, template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > self.confidence_threshold and max_val > best_confidence:
                    template_h, template_w = template.shape
                    x, y = max_loc

                    best_match = {
                        "state": state,
                        "bbox": {
                            "x": x,
                            "y": y,
                            "width": template_w,
                            "height": template_h,
                            "x2": x + template_w,
                            "y2": y + template_h,
                        },
                        "confidence": float(max_val),
                        "center_x": x + template_w // 2,
                        "center_y": y + template_h // 2,
                    }
                    best_confidence = max_val

            return best_match

        except Exception as e:
            self.logger.error("Ошибка детекции состояния T-Rex", e)
            return None

    def _detect_template_objects(
        self, gray_image: np.ndarray, obj_type: str, template: np.ndarray
    ) -> List[Dict]:
        objects = []

        try:
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.confidence_threshold)

            template_h, template_w = template.shape

            for pt in zip(*locations[::-1]):
                x, y = pt
                confidence = result[y, x]

                bbox = {
                    "x": x,
                    "y": y,
                    "width": template_w,
                    "height": template_h,
                    "x2": x + template_w,
                    "y2": y + template_h,
                }

                objects.append(
                    {
                        "type": obj_type,
                        "bbox": bbox,
                        "confidence": float(confidence),
                        "is_threat": self._is_threat(obj_type),
                        "center_x": x + template_w // 2,
                        "center_y": y + template_h // 2,
                    }
                )

        except Exception as e:
            self.logger.error(f"Ошибка детекции шаблона {obj_type}", e)

        return objects

    def _detect_pterodactyl(self, gray_image: np.ndarray) -> List[Dict]:
        objects = []

        try:

            if "pterodactyl" in self.templates:
                return objects

            _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                if 200 < area < 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0

                    if 1.5 < aspect_ratio < 4.0:
                        roi = thresh[y : y + h, x : x + w]
                        if roi.size > 0:
                            density = cv2.countNonZero(roi) / (w * h)

                            if 0.3 < density < 0.8:
                                bbox = {
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h,
                                    "x2": x + w,
                                    "y2": y + h,
                                }

                                objects.append(
                                    {
                                        "type": "pterodactyl_detected",
                                        "bbox": bbox,
                                        "confidence": float(density),
                                        "is_threat": True,
                                        "center_x": x + w // 2,
                                        "center_y": y + h // 2,
                                    }
                                )

        except Exception as e:
            self.logger.error("Ошибка детекции птеродактиля", e)

        return objects

    def _filter_overlapping_objects(self, objects: List[Dict]) -> List[Dict]:
        if not objects:
            return objects

        sorted_objects = sorted(objects, key=lambda x: x["confidence"], reverse=True)
        filtered_objects = []

        for obj in sorted_objects:
            is_overlapping = False

            for filtered_obj in filtered_objects:
                if self._calculate_iou(obj["bbox"], filtered_obj["bbox"]) > 0.3:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered_objects.append(obj)

        return filtered_objects

    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        try:
            x1 = max(bbox1["x"], bbox2["x"])
            y1 = max(bbox1["y"], bbox2["y"])
            x2 = min(bbox1["x2"], bbox2["x2"])
            y2 = min(bbox1["y2"], bbox2["y2"])

            if x2 <= x1 or y2 <= y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)

            area1 = bbox1["width"] * bbox1["height"]
            area2 = bbox2["width"] * bbox2["height"]

            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _is_threat(self, obj_type: str) -> bool:

        safe_objects = {
            "trex",
            "dead_trex",
            "trex_in_jump",
            "canvas",
        }

        if obj_type in safe_objects:
            return False

        return True

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = max(0.1, min(0.99, threshold))
        self.logger.info(f"Порог уверенности установлен: {self.confidence_threshold}")

    def get_template_info(self) -> Dict[str, Dict]:
        info = {}
        for obj_type, template in self.templates.items():
            info[obj_type] = {
                "shape": template.shape,
                "is_threat": self._is_threat(obj_type),
                "loaded": True,
            }
        return info
