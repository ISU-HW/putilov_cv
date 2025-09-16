import cv2
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple
from logger import Logger


class ObjectDetector:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.templates: Dict[str, np.ndarray] = {}
        self.confidence_threshold = 0.6
        self.low_confidence_threshold = 0.4

        
        self.screenshot_counter = 0

        
        self.last_screenshot_time = 0
        self.screenshot_interval = 1.0  

        
        os.makedirs("debug_screenshots", exist_ok=True)

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

    def should_save_screenshot(self) -> bool:
        current_time = time.time()
        if current_time - self.last_screenshot_time >= self.screenshot_interval:
            self.last_screenshot_time = current_time
            return True
        return False

    def save_obstacle_screenshot(
        self,
        original_screenshot: np.ndarray,
        detected_objects: List[Dict],
        trex_info: Optional[Dict] = None,
    ) -> None:
        try:
            
            if not self.should_save_screenshot():
                return

            if not detected_objects:
                return

            
            if (
                len(original_screenshot.shape) == 3
                and original_screenshot.shape[2] == 3
            ):
                
                screenshot_bgr = cv2.cvtColor(original_screenshot, cv2.COLOR_RGB2BGR)
            else:
                screenshot_bgr = original_screenshot.copy()

            
            obstacles_by_type = {}
            for obj in detected_objects:
                obj_type = obj["type"]
                if obj_type not in obstacles_by_type:
                    obstacles_by_type[obj_type] = []
                obstacles_by_type[obj_type].append(obj)

            
            all_obstacles_screenshot = screenshot_bgr.copy()

            
            if trex_info:
                x, y, w, h = (
                    trex_info["x"],
                    trex_info["y"],
                    trex_info["width"],
                    trex_info["height"],
                )
                cv2.rectangle(
                    all_obstacles_screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2
                )  
                cv2.putText(
                    all_obstacles_screenshot,
                    f"T-Rex {trex_info['state']}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            for obj_type, objects in obstacles_by_type.items():
                
                type_screenshot = screenshot_bgr.copy()

                
                if trex_info:
                    x, y, w, h = (
                        trex_info["x"],
                        trex_info["y"],
                        trex_info["width"],
                        trex_info["height"],
                    )
                    cv2.rectangle(
                        type_screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )  
                    cv2.putText(
                        type_screenshot,
                        f"T-Rex {trex_info['state']}",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                for obj in objects:
                    
                    x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
                    confidence = obj["confidence"]

                    
                    color = (0, 0, 255)  
                    thickness = 2

                    
                    cv2.rectangle(
                        type_screenshot, (x, y), (x + w, y + h), color, thickness
                    )
                    cv2.rectangle(
                        all_obstacles_screenshot,
                        (x, y),
                        (x + w, y + h),
                        color,
                        thickness,
                    )

                    
                    label = f"{obj_type} {confidence:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1

                    
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, font, font_scale, font_thickness
                    )

                    
                    cv2.rectangle(
                        type_screenshot,
                        (x, y - text_height - 5),
                        (x + text_width, y),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.rectangle(
                        all_obstacles_screenshot,
                        (x, y - text_height - 5),
                        (x + text_width, y),
                        (0, 0, 0),
                        -1,
                    )

                    
                    cv2.putText(
                        type_screenshot,
                        label,
                        (x, y - 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                    )
                    cv2.putText(
                        all_obstacles_screenshot,
                        label,
                        (x, y - 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                    )

                
                type_filename = f"debug_screenshots/obstacle_{obj_type}.png"
                cv2.imwrite(type_filename, type_screenshot)
                self.logger.info(
                    f"Сохранен скриншот препятствия {obj_type}: {type_filename} (найдено объектов: {len(objects)})"
                )

            
            self.screenshot_counter += 1
            all_filename = (
                f"debug_screenshots/all_obstacles_{self.screenshot_counter:04d}.png"
            )
            cv2.imwrite(all_filename, all_obstacles_screenshot)
            self.logger.info(f"Сохранен общий скриншот с препятствиями: {all_filename}")

        except Exception as e:
            self.logger.error("Ошибка при сохранении скриншота препятствий", e)

    def _calculate_exclusion_zone(self, trex_info: Dict) -> Dict:
        try:
            
            padding_x = int(trex_info["width"] * 0.5)  
            padding_y = int(trex_info["height"] * 0.3)  

            exclusion_zone = {
                "x1": max(0, trex_info["x"] - padding_x),
                "y1": max(0, trex_info["y"] - padding_y),
                "x2": trex_info["x"] + trex_info["width"] + padding_x,
                "y2": trex_info["y"] + trex_info["height"] + padding_y,
            }

            return exclusion_zone

        except Exception as e:
            self.logger.error("Ошибка при расчете зоны исключения", e)
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

    def _is_in_exclusion_zone(
        self,
        obj_x: int,
        obj_y: int,
        obj_width: int,
        obj_height: int,
        exclusion_zone: Dict,
    ) -> bool:
        obj_center_x = obj_x + obj_width // 2
        obj_center_y = obj_y + obj_height // 2

        return (
            exclusion_zone["x1"] <= obj_center_x <= exclusion_zone["x2"]
            and exclusion_zone["y1"] <= obj_center_y <= exclusion_zone["y2"]
        )

    def _validate_pterodactyl_position(
        self, obj: Dict, screenshot_shape: Tuple
    ) -> bool:
        try:
            img_height, img_width = screenshot_shape[:2]

            
            
            ground_threshold = img_height * 0.6

            if obj["center_y"] > ground_threshold:
                self.logger.debug(
                    f"Птеродактиль отклонен: слишком низко (y={obj['center_y']}, порог={ground_threshold})"
                )
                return False

            
            max_width = img_width * 0.15  
            max_height = img_height * 0.2  

            if obj["width"] > max_width or obj["height"] > max_height:
                self.logger.debug(
                    f"Птеродактиль отклонен: слишком большой ({obj['width']}x{obj['height']})"
                )
                return False

            return True

        except Exception as e:
            self.logger.error("Ошибка при валидации позиции птеродактиля", e)
            return False

    def detect_objects(
        self, screenshot: np.ndarray, trex_info: Optional[Dict] = None
    ) -> List[Dict]:
        detected_objects = []

        try:
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            
            gray_screenshot = cv2.GaussianBlur(gray_screenshot, (3, 3), 0)

            
            exclusion_zone = None
            if trex_info:
                exclusion_zone = self._calculate_exclusion_zone(trex_info)
                
                

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
                        gray_screenshot,
                        obj_type,
                        self.templates[obj_type],
                        exclusion_zone,
                        screenshot.shape,
                    )
                    detected_objects.extend(objects)

            detected_objects = self._filter_overlapping_objects(detected_objects)
            detected_objects = self._classify_pterodactyl_heights(detected_objects)

            
            if detected_objects:
                self.save_obstacle_screenshot(screenshot, detected_objects, trex_info)

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

            
            regions = [
                gray_screenshot,  
                gray_screenshot[:100, :],  
                gray_screenshot[25:125, :],  
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
        self,
        gray_image: np.ndarray,
        obj_type: str,
        template: np.ndarray,
        exclusion_zone: Optional[Dict] = None,
        screenshot_shape: Optional[Tuple] = None,
    ) -> List[Dict]:
        objects = []

        try:
            img_h, img_w = gray_image.shape
            template_h, template_w = template.shape

            if template_h > img_h or template_w > img_w:
                return objects

            
            threshold = self.confidence_threshold
            if "pterodactyl" in obj_type:
                threshold = 0.65  
            elif "cactus" in obj_type:
                threshold = 0.65  

            
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

                    
                    if "pterodactyl" in obj_type and exclusion_zone:
                        if self._is_in_exclusion_zone(
                            x, y, scaled_w, scaled_h, exclusion_zone
                        ):
                            
                            
                            continue

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
                        if (
                            screenshot_shape
                            and not self._validate_pterodactyl_position(
                                obj_data, screenshot_shape
                            )
                        ):
                            continue
                        obj_data["is_flying"] = True

                    objects.append(obj_data)

        except Exception as e:
            self.logger.error(f"Error detecting template {obj_type}", e)

        return objects

    def _classify_pterodactyl_heights(self, objects: List[Dict]) -> List[Dict]:
        for obj in objects:
            if "pterodactyl" in obj["type"]:
                y_pos = obj["center_y"]

                
                if y_pos < 40:
                    obj["height_level"] = "high"
                    obj["duck_required"] = False
                elif y_pos < 70:
                    obj["height_level"] = "medium"
                    obj["duck_required"] = True  
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
