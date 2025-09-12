import mss
import numpy as np
import cv2
import threading
import time
import os
import keyboard
import pyautogui
import json
from typing import Dict, Optional
from debug import Logger
from detector import ObjectDetector
from calculate import TrajectoryCalculator
from enum import Enum


class BotState(Enum):
    READY = "ready"
    SEARCHING_DINO = "searching_dino"
    DINO_FOUND = "dino_found"
    DINO_NOT_FOUND = "dino_not_found"
    WAITING_FOR_GAME = "waiting_for_game"
    PLAYING = "playing"
    GAME_OVER = "game_over"
    STOPPED = "stopped"
    ERROR = "error"


class GameStats:
    def __init__(self):
        self.current_game_time = 0
        self.current_jumps = 0
        self.current_score = 0
        self.is_game_over = False
        self.game_over_reason = ""


class TRexBot:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.is_running: bool = False
        self.bot_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.capture_area: Optional[Dict] = None
        self.dino_template: Optional[np.ndarray] = None

        self.state = BotState.READY
        self.status_message = "Готов к работе"
        self.error_message = ""

        self.game_stats = GameStats()

        self.object_detector = ObjectDetector(logger)
        self.trajectory_calculator = TrajectoryCalculator(logger)

        self.last_detected_objects = []
        self.last_trajectory = []
        self.last_dino_info = None

        self.default_settings: Dict[str, float] = {
            "jump_delay": 0.1,
            "scan_delay": 0.01,
            "confidence_threshold": 0.7,
            "obstacle_density": 0.05,
            "capture_width": 600,
            "capture_height": 150,
            "gameover_confidence": 0.8,
        }

        self.settings: Dict[str, float] = self.default_settings.copy()
        self.stats: Dict[str, float] = {
            "games_played": 0,
            "best_score": 0,
            "total_time": 0,
            "total_jumps": 0,
        }

        self.initialize_files()
        self.validate_and_fix_settings()

    def validate_and_fix_settings(self) -> None:
        try:
            self.settings["capture_width"] = max(
                100, int(self.settings["capture_width"])
            )
            self.settings["capture_height"] = max(
                50, int(self.settings["capture_height"])
            )

            self.settings["jump_delay"] = max(
                0.01, min(1.0, float(self.settings["jump_delay"]))
            )
            self.settings["scan_delay"] = max(
                0.001, min(0.5, float(self.settings["scan_delay"]))
            )
            self.settings["confidence_threshold"] = max(
                0.1, min(0.99, float(self.settings["confidence_threshold"]))
            )
            self.settings["obstacle_density"] = max(
                0.001, min(0.5, float(self.settings["obstacle_density"]))
            )

            self.logger.info("Настройки проверены и исправлены")

        except Exception as e:
            self.logger.error("Ошибка валидации настроек", e)
            self.settings = self.default_settings.copy()
            self.logger.warning("Настройки сброшены к значениям по умолчанию")

    def initialize_files(self) -> None:
        try:
            os.makedirs("images", exist_ok=True)
            self.logger.info("Папка images создана/проверена")

            if not os.path.exists("images/canvas.png"):
                self.logger.warning("Файл images/canvas.png не найден!")
                raise FileNotFoundError("Шаблон динозавра не найден")
            else:
                self.load_dino_template()

            if os.path.exists("images/gameover.png"):
                self.gameover_template = cv2.imread(
                    "images/gameover.png", cv2.IMREAD_GRAYSCALE
                )
                if self.gameover_template is not None:
                    self.logger.info(
                        f"Шаблон 'Game Over' загружен: {self.gameover_template.shape}"
                    )
                else:
                    self.logger.warning("Не удалось загрузить шаблон 'Game Over'")
            else:
                self.logger.warning("Файл images/gameover.png не найден!")
                self.gameover_template = None

            if os.path.exists("trex_settings.json"):
                self.load_settings()
            else:
                self.save_settings()

            if os.path.exists("trex_stats.json"):
                self.load_stats()
            else:
                self.save_stats()

        except Exception as e:
            self.logger.error("Ошибка инициализации файлов", e)
            self.state = BotState.ERROR
            self.error_message = f"Ошибка инициализации: {str(e)}"

    def load_dino_template(self) -> None:
        try:
            self.dino_template = cv2.imread("images/canvas.png", cv2.IMREAD_GRAYSCALE)
            if self.dino_template is not None:
                self.logger.info(
                    f"Шаблон динозавра загружен: {self.dino_template.shape}"
                )
            else:
                raise ValueError("Не удалось загрузить изображение")
        except Exception as e:
            self.logger.error("Ошибка загрузки шаблона динозавра", e)

    def find_dino_position(self, screenshot: np.ndarray) -> Optional[Dict]:
        if self.dino_template is None:
            self.logger.warning("Шаблон динозавра не загружен")
            return None

        try:
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(
                gray_screenshot, self.dino_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > self.settings["confidence_threshold"]:
                x, y = max_loc
                template_h, _ = self.dino_template.shape
                position = {"x": x, "y": y + template_h, "confidence": max_val}
                return position
        except Exception as e:
            self.logger.error("Ошибка поиска динозавра", e)
        return None

    def detect_gameover(self, screenshot: np.ndarray) -> bool:
        if self.gameover_template is None:
            return False

        try:
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(
                gray_screenshot, self.gameover_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            return max_val > self.settings["gameover_confidence"]
        except Exception as e:
            self.logger.error("Ошибка детекции gameover", e)
            return False

    def calculate_capture_area(self, dino_pos: Dict, monitor: Dict) -> Dict:
        area = {
            "top": int(dino_pos["y"] - self.settings["capture_height"]),
            "left": int(dino_pos["x"]),
            "width": int(self.settings["capture_width"]),
            "height": int(self.settings["capture_height"]),
        }

        area["top"] = max(0, area["top"])
        area["left"] = max(0, area["left"])
        area["width"] = max(1, area["width"])
        area["height"] = max(1, area["height"])

        return area

    def start_bot(self) -> None:
        if self.is_running:
            return

        try:
            self.is_running = True
            self.state = BotState.SEARCHING_DINO
            self.status_message = "Запуск бота..."
            self.game_stats = GameStats()

            self.logger.info("Запуск бота")
            self.bot_thread = threading.Thread(target=self.run_bot, daemon=True)
            self.bot_thread.start()
        except Exception as e:
            self.logger.error("Ошибка запуска бота", e)
            self.state = BotState.ERROR
            self.error_message = f"Ошибка запуска: {str(e)}"
            self.is_running = False

    def stop_bot(self) -> None:
        try:
            self.is_running = False
            self.state = BotState.STOPPED
            self.status_message = "Остановлен"
            self.logger.info("Бот остановлен")
        except Exception as e:
            self.logger.error("Ошибка остановки бота", e)

    def run_bot(self) -> None:
        try:
            self.logger.info("Начинаем поиск экранов")
            self.state = BotState.SEARCHING_DINO
            self.status_message = "Поиск динозавра..."

            search_result = self._capture_screens()
            if not search_result:
                self.logger.error("Динозавр не найден ни на одном экране")
                self.state = BotState.DINO_NOT_FOUND
                self.status_message = "Дино не найден"
                return

            self.capture_area = self.calculate_capture_area(
                search_result["dino_pos"], search_result["monitor"]
            )

            self.logger.info("Динозавр найден, ожидание запуска игры")
            self.state = BotState.DINO_FOUND
            self.status_message = "Дино найден"

            self.state = BotState.WAITING_FOR_GAME
            self.status_message = "Нажмите ПРОБЕЛ для начала игры"

            while self.is_running and not keyboard.is_pressed("space"):
                time.sleep(0.1)

            if not self.is_running:
                return

            self.start_time = time.time()
            self.game_stats = GameStats()
            self.logger.info("Игра началась")
            self.state = BotState.PLAYING
            self.status_message = "Играет"

            while self.is_running:
                if keyboard.is_pressed("esc"):
                    self.logger.info("Выход по ESC")
                    break

                try:
                    with mss.mss() as sct:
                        screenshot = np.array(sct.grab(self.capture_area))

                    self.game_stats.current_game_time = (
                        time.time() - self.start_time if self.start_time else 0
                    )
                    self.game_stats.current_score = int(
                        self.game_stats.current_game_time * 10
                    )

                    if self.detect_gameover(screenshot):
                        self.handle_game_over()
                        break

                    detected_objects = self.object_detector.detect_objects(screenshot)
                    self.last_detected_objects = detected_objects

                    trex_position = (50, 93)

                    threats = [
                        obj for obj in detected_objects if obj.get("is_threat", False)
                    ]

                    if threats:

                        should_jump, reason = self.trajectory_calculator.should_jump(
                            trex_position, threats, self.game_stats.current_game_time
                        )

                        if should_jump:
                            trajectory = (
                                self.trajectory_calculator.calculate_jump_trajectory(
                                    trex_position[0],
                                    trex_position[1],
                                    self.game_stats.current_game_time,
                                )
                            )
                            self.last_trajectory = trajectory
                        else:
                            self.last_trajectory = []

                        if should_jump:
                            pyautogui.press("space")
                            self.game_stats.current_jumps += 1
                            self.logger.info(
                                f"Прыжок №{self.game_stats.current_jumps}: {reason}"
                            )
                            time.sleep(self.settings["jump_delay"])

                    time.sleep(self.settings["scan_delay"])

                except Exception as e:
                    self.logger.error(f"Ошибка в основном цикле бота: {str(e)}", e)
                    time.sleep(0.5)

        except Exception as e:
            self.logger.error("Критическая ошибка в основном цикле бота", e)
            self.state = BotState.ERROR
            self.error_message = f"Ошибка: {str(e)}"
        finally:
            if self.is_running:
                self.stop_bot()

    def handle_game_over(self):
        try:
            self.logger.info("Обнаружен экран 'Game Over'")

            self.stats.update(
                {
                    "games_played": self.stats["games_played"] + 1,
                    "total_time": self.stats["total_time"]
                    + self.game_stats.current_game_time,
                    "total_jumps": self.stats["total_jumps"]
                    + self.game_stats.current_jumps,
                    "best_score": max(
                        self.stats["best_score"], self.game_stats.current_score
                    ),
                }
            )

            self.logger.info(
                f"Игра завершена. Время: {self.game_stats.current_game_time:.1f}с, "
                f"Прыжков: {self.game_stats.current_jumps}, Счёт: {self.game_stats.current_score}"
            )

            self.state = BotState.GAME_OVER
            self.status_message = (
                f"Игра окончена. Счёт: {self.game_stats.current_score}"
            )
            self.game_stats.is_game_over = True
            self.save_stats()

        except Exception as e:
            self.logger.error("Ошибка обработки завершения игры", e)

    def restart_game(self):
        try:
            pyautogui.press("space")
            self.game_stats = GameStats()
            self.start_time = time.time()
            self.state = BotState.PLAYING
            self.status_message = "Играет"
            self.logger.info("Игра перезапущена")
        except Exception as e:
            self.logger.error("Ошибка перезапуска игры", e)

    def _capture_screens(self) -> Optional[Dict]:
        try:
            with mss.mss() as sct:
                monitors = sct.monitors[1:]
                self.logger.info(f"Найдено {len(monitors)} мониторов")

                for i, monitor in enumerate(monitors, 1):
                    self.status_message = f"Поиск динозавра на экране {i}..."
                    screenshot = np.array(sct.grab(monitor))[:, :, :3]
                    dino_pos = self.find_dino_position(screenshot)
                    if dino_pos:
                        dino_pos["x"] += monitor["left"]
                        dino_pos["y"] += monitor["top"]
                        self.logger.info(f"Динозавр найден на мониторе {i}")
                        return {
                            "dino_pos": dino_pos,
                            "monitor": monitor,
                            "screenshot": screenshot,
                        }
                self.logger.warning("Динозавр не найден ни на одном мониторе")
                return None
        except Exception as e:
            self.logger.error("Ошибка захвата экранов", e)
            self.state = BotState.ERROR
            self.error_message = f"Ошибка захвата: {str(e)}"
            return None

    def get_debug_info(self) -> Dict:
        return {
            "capture_area": self.capture_area,
            "detected_objects": self.last_detected_objects,
            "trajectory": self.last_trajectory,
            "dino_info": self.last_dino_info,
            "should_show": self.state
            in [BotState.DINO_FOUND, BotState.WAITING_FOR_GAME, BotState.PLAYING],
        }

    def save_settings(self) -> None:
        try:
            with open("trex_settings.json", "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            self.logger.info("Настройки сохранены")
        except Exception as e:
            self.logger.error("Ошибка сохранения настроек", e)

    def load_settings(self) -> None:
        try:
            with open("trex_settings.json", "r", encoding="utf-8") as f:
                loaded_settings = json.load(f)
                self.settings.update(loaded_settings)
            self.validate_and_fix_settings()
            self.logger.info("Настройки загружены")
        except Exception as e:
            self.logger.error("Ошибка загрузки настроек", e)

    def save_stats(self) -> None:
        try:
            with open("trex_stats.json", "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error("Ошибка сохранения статистики", e)

    def load_stats(self) -> None:
        try:
            with open("trex_stats.json", "r", encoding="utf-8") as f:
                loaded_stats = json.load(f)
                self.stats.update(loaded_stats)
            self.logger.info("Статистика загружена")
        except Exception as e:
            self.logger.error("Ошибка загрузки статистики", e)

    def reset_stats(self) -> None:
        try:
            self.stats = {
                "games_played": 0,
                "best_score": 0,
                "total_time": 0,
                "total_jumps": 0,
            }
            self.save_stats()
            self.logger.info("Статистика сброшена")
        except Exception as e:
            self.logger.error("Ошибка сброса статистики", e)
