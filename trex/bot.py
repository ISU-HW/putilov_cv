import mss
import numpy as np
import cv2
import threading
import time
import os
import keyboard
import pyautogui
import json
from typing import Dict, List, Optional
from enum import Enum

from logger import Logger
from detector import ObjectDetector
from trajectory import TrajectoryCalculator


class BotState(Enum):
    READY = "ready"
    SEARCHING_CANVAS = "searching_canvas"
    CANVAS_FOUND = "canvas_found"
    CANVAS_NOT_FOUND = "canvas_not_found"
    WAITING_FOR_GAME = "waiting_for_game"
    PLAYING = "playing"
    GAME_OVER = "game_over"
    STOPPED = "stopped"
    ERROR = "error"


class GameStats:
    def __init__(self):
        self.current_game_time = 0.0
        self.current_jumps = 0
        self.current_ducks = 0
        self.current_score = 0
        self.is_game_over = False
        self.actions_taken = []


class TRexBot:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.is_running = False
        self.bot_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.capture_area: Optional[Dict] = None
        self.canvas_template: Optional[np.ndarray] = None
        self.canvas_width: int = 0
        self.canvas_height: int = 0

        self.state = BotState.READY
        self.status_message = "Ready to start"
        self.error_message = ""
        self.game_stats = GameStats()

        self.object_detector = ObjectDetector(logger)
        self.trajectory_calculator = TrajectoryCalculator(logger)

        # Отслеживание действий
        self.last_action_time = 0
        self.action_cooldown = 0.05
        self.duck_duration = 0.3
        self.is_ducking = False
        self.duck_start_time = 0

        self.settings = {
            "jump_delay": 0.1,
            "scan_delay": 0.01,
            "confidence_threshold": 0.6,
            "jump_sensitivity": 0.05,
            "gameover_confidence": 0.7,
            "duck_duration": 0.3,
        }

        self.stats = {
            "games_played": 0,
            "best_score": 0,
            "total_time": 0,
            "total_jumps": 0,
            "total_ducks": 0,
        }

        self.initialize_files()

    def initialize_files(self) -> None:
        try:
            os.makedirs("images", exist_ok=True)
            os.makedirs("debug_screenshots", exist_ok=True)
            self.logger.info("Images and debug directories created/verified")

            if not os.path.exists("images/canvas.png"):
                self.logger.error("Canvas template (canvas.png) not found!")
                self.state = BotState.ERROR
                self.error_message = "Canvas template not found"
                return
            else:
                self.load_canvas_template()

            if os.path.exists("trex_settings.json"):
                self.load_settings()
            else:
                self.save_settings()

            if os.path.exists("trex_stats.json"):
                self.load_stats()
            else:
                self.save_stats()

        except Exception as e:
            self.logger.error("Error initializing files", e)
            self.state = BotState.ERROR
            self.error_message = f"Initialization error: {str(e)}"

    def load_canvas_template(self) -> None:
        try:
            self.canvas_template = cv2.imread("images/canvas.png", cv2.IMREAD_GRAYSCALE)
            if self.canvas_template is not None:
                self.canvas_height, self.canvas_width = self.canvas_template.shape
                self.logger.info(
                    f"Canvas template loaded successfully: {self.canvas_width}x{self.canvas_height}"
                )

                # Сохраняем отладочную информацию о template
                debug_filename = "debug_screenshots/canvas_template.png"
                cv2.imwrite(debug_filename, self.canvas_template)
                self.logger.info(f"Canvas template saved to: {debug_filename}")

            else:
                raise ValueError("Failed to load canvas template image")
        except Exception as e:
            self.logger.error("Error loading canvas template", e)
            self.canvas_template = None

    def find_canvas_area(self, screenshot: np.ndarray) -> Optional[Dict]:
        if not hasattr(self, "canvas_template") or self.canvas_template is None:
            self.logger.error("Canvas template not loaded")
            return None

        try:
            gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            img_h, img_w = gray_screenshot.shape
            template_h, template_w = self.canvas_template.shape

            self.logger.info(f"Screenshot size: {img_w}x{img_h}")
            self.logger.info(f"Canvas template size: {template_w}x{template_h}")

            if template_h > img_h or template_w > img_w:
                self.logger.warning(
                    f"Canvas template ({template_w}x{template_h}) is larger than image ({img_w}x{img_h})"
                )
                return None

            # Ищем точное совпадение canvas.png на экране
            result = cv2.matchTemplate(
                gray_screenshot, self.canvas_template, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            self.logger.info(
                f"Template matching result: confidence={max_val:.3f}, location={max_loc}"
            )

            # Попробуем разные пороги
            thresholds = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

            for threshold in thresholds:
                if max_val > threshold:
                    x, y = max_loc
                    self.logger.info(
                        f"Canvas found with threshold {threshold} at ({x}, {y})"
                    )
                    return {
                        "x": x,
                        "y": y,
                        "width": template_w,
                        "height": template_h,
                        "confidence": max_val,
                    }

            self.logger.warning(
                f"Canvas not found with any threshold. Best match: {max_val:.3f} at {max_loc}"
            )

        except Exception as e:
            self.logger.error("Error finding canvas area", e)

        return None

    def set_manual_capture_area(self, x: int, y: int, width: int, height: int) -> None:
        """Ручная установка области захвата для отладки"""
        self.capture_area = {"top": y, "left": x, "width": width, "height": height}
        self.logger.info(f"Manual capture area set: {self.capture_area}")

    def test_canvas_matching(self) -> None:
        """Тестирует поиск canvas на всех мониторах"""
        try:
            with mss.mss() as sct:
                for i, monitor in enumerate(sct.monitors[1:], 1):
                    screenshot = np.array(sct.grab(monitor))[:, :, :3]

                    debug_filename = f"debug_screenshots/test_monitor_{i}.png"
                    cv2.imwrite(
                        debug_filename, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                    )

                    canvas_area = self.find_canvas_area(screenshot)
                    self.logger.info(f"Monitor {i}: canvas_area = {canvas_area}")

        except Exception as e:
            self.logger.error("Error in test_canvas_matching", e)

    def start_bot(self) -> None:
        if self.is_running:
            return

        try:
            self.is_running = True
            self.state = BotState.SEARCHING_CANVAS
            self.status_message = "Starting bot..."
            self.game_stats = GameStats()

            self.logger.info("=== STARTING NEW CANVAS-BASED BOT ===")
            self.bot_thread = threading.Thread(target=self.run_bot, daemon=True)
            self.bot_thread.start()

        except Exception as e:
            self.logger.error("Error starting bot", e)
            self.state = BotState.ERROR
            self.error_message = f"Start error: {str(e)}"
            self.is_running = False

    def stop_bot(self) -> None:
        try:
            self.is_running = False
            self.state = BotState.STOPPED
            self.status_message = "Stopped"

            # Завершаем пригибание если активно
            if self.is_ducking:
                pyautogui.keyUp("down")
                self.is_ducking = False

            self.logger.info("Bot stopped")
        except Exception as e:
            self.logger.error("Error stopping bot", e)

    def run_bot(self) -> None:
        try:
            self.logger.info("=== SEARCHING FOR GAME CANVAS ===")
            self.state = BotState.SEARCHING_CANVAS
            self.status_message = "Searching for game canvas..."

            search_result = self._capture_screens()
            if not search_result:
                self.logger.error("Game canvas not found on any screen")
                self.state = BotState.CANVAS_NOT_FOUND
                self.status_message = "Game canvas not found"
                return

            self.capture_area = search_result["capture_area"]
            self.logger.info(f"=== CAPTURE AREA SET TO: {self.capture_area} ===")

            self.state = BotState.CANVAS_FOUND
            self.status_message = "Game canvas found, press SPACE to start"

            self.state = BotState.WAITING_FOR_GAME
            while self.is_running and not keyboard.is_pressed("space"):
                time.sleep(0.1)

            if not self.is_running:
                return

            self.start_time = time.time()
            self.game_stats = GameStats()
            self.logger.info("Game started")
            self.state = BotState.PLAYING
            self.status_message = "Playing"

            while self.is_running:
                if keyboard.is_pressed("esc"):
                    self.logger.info("Exit requested (ESC)")
                    break

                try:
                    with mss.mss() as sct:
                        screenshot = np.array(sct.grab(self.capture_area))

                    if screenshot.shape[1] < 50 or screenshot.shape[0] < 50:
                        self.logger.error(f"Screenshot too small: {screenshot.shape}")
                        time.sleep(0.5)
                        continue

                    # Отладочные скриншоты
                    frame_count = getattr(self, "frame_count", 0)
                    self.frame_count = frame_count + 1

                    if self.frame_count <= 5 or self.frame_count % 100 == 0:
                        debug_filename = (
                            f"debug_screenshots/game_frame_{self.frame_count}.png"
                        )
                        cv2.imwrite(
                            debug_filename, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                        )
                        self.logger.info(
                            f"Saved game screenshot: {debug_filename} (shape: {screenshot.shape})"
                        )

                    # Обновляем статистику
                    self.game_stats.current_game_time = (
                        time.time() - self.start_time if self.start_time else 0
                    )
                    self.game_stats.current_score = int(
                        self.game_stats.current_game_time * 10
                    )

                    # Проверяем Game Over
                    if self.object_detector.detect_game_over(screenshot):
                        self.handle_game_over()
                        break

                    # Обнаруживаем T-Rex
                    trex_info = self.object_detector.detect_trex(screenshot)

                    if trex_info:
                        # Обнаруживаем препятствия
                        obstacles = self.object_detector.detect_objects(screenshot)

                        if obstacles:
                            action, reason = (
                                self.trajectory_calculator.should_jump_or_duck(
                                    trex_info,
                                    obstacles,
                                    self.game_stats.current_game_time,
                                )
                            )

                            if action != "none":
                                self.execute_action(action, reason)

                    # Обрабатываем продолжающееся пригибание
                    self.handle_ducking()

                    time.sleep(self.settings["scan_delay"])

                except Exception as e:
                    self.logger.error(f"Error in main game loop: {str(e)}", e)
                    time.sleep(0.5)

        except Exception as e:
            self.logger.error("Critical error in bot main loop", e)
            self.state = BotState.ERROR
            self.error_message = f"Error: {str(e)}"
        finally:
            if self.is_running:
                self.stop_bot()

    def execute_action(self, action: str, reason: str) -> None:
        try:
            current_time = time.time()

            if current_time - self.last_action_time < self.action_cooldown:
                return

            if action == "jump":
                pyautogui.press("space")
                self.game_stats.current_jumps += 1
                self.logger.info(f"Jump #{self.game_stats.current_jumps}: {reason}")

            elif action == "duck":
                pyautogui.keyDown("down")
                self.is_ducking = True
                self.duck_start_time = current_time
                self.game_stats.current_ducks += 1
                self.logger.info(f"Duck #{self.game_stats.current_ducks}: {reason}")

            self.last_action_time = current_time
            self.game_stats.actions_taken.append(
                {
                    "action": action,
                    "time": self.game_stats.current_game_time,
                    "reason": reason,
                }
            )

        except Exception as e:
            self.logger.error(f"Error executing action {action}", e)

    def handle_ducking(self) -> None:
        if self.is_ducking:
            current_time = time.time()
            if current_time - self.duck_start_time >= self.settings["duck_duration"]:
                try:
                    pyautogui.keyUp("down")
                    self.is_ducking = False
                except Exception as e:
                    self.logger.error("Error ending duck", e)
                    self.is_ducking = False

    def handle_game_over(self):
        try:
            self.logger.info("Game Over detected")

            if self.is_ducking:
                pyautogui.keyUp("down")
                self.is_ducking = False

            self.stats.update(
                {
                    "games_played": self.stats["games_played"] + 1,
                    "total_time": self.stats["total_time"]
                    + self.game_stats.current_game_time,
                    "total_jumps": self.stats["total_jumps"]
                    + self.game_stats.current_jumps,
                    "total_ducks": self.stats["total_ducks"]
                    + self.game_stats.current_ducks,
                    "best_score": max(
                        self.stats["best_score"], self.game_stats.current_score
                    ),
                }
            )

            self.logger.info(
                f"Game finished. Time: {self.game_stats.current_game_time:.1f}s, "
                f"Jumps: {self.game_stats.current_jumps}, "
                f"Ducks: {self.game_stats.current_ducks}, "
                f"Score: {self.game_stats.current_score}"
            )

            self.state = BotState.GAME_OVER
            self.status_message = f"Game Over. Score: {self.game_stats.current_score}"
            self.game_stats.is_game_over = True
            self.save_stats()

        except Exception as e:
            self.logger.error("Error handling game over", e)

    def restart_game(self):
        try:
            if self.is_ducking:
                pyautogui.keyUp("down")
                self.is_ducking = False

            pyautogui.press("space")

            self.game_stats = GameStats()
            self.start_time = time.time()
            self.state = BotState.PLAYING
            self.status_message = "Playing"
            self.logger.info("Game restarted")

        except Exception as e:
            self.logger.error("Error restarting game", e)

    def _capture_screens(self) -> Optional[Dict]:
        try:
            with mss.mss() as sct:
                monitors = sct.monitors[1:]
                self.logger.info(f"Found {len(monitors)} monitors")

                for i, monitor in enumerate(monitors, 1):
                    self.status_message = f"Searching canvas on screen {i}..."
                    self.logger.info(f"Monitor {i}: {monitor}")

                    screenshot = np.array(sct.grab(monitor))[:, :, :3]

                    # Сохраняем отладочный скриншот полного экрана
                    debug_filename = f"debug_screenshots/monitor_{i}_full.png"
                    cv2.imwrite(
                        debug_filename, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                    )
                    self.logger.info(f"Saved debug screenshot: {debug_filename}")

                    canvas_area = self.find_canvas_area(screenshot)

                    if canvas_area:
                        # Преобразуем в глобальные координаты
                        global_canvas_area = {
                            "top": canvas_area["y"] + monitor["top"],
                            "left": canvas_area["x"] + monitor["left"],
                            "width": canvas_area["width"],
                            "height": canvas_area["height"],
                        }

                        self.logger.info(f"Canvas found on monitor {i}")
                        self.logger.info(f"Local canvas area: {canvas_area}")
                        self.logger.info(f"Global capture area: {global_canvas_area}")

                        # Отмечаем найденную область
                        debug_screenshot = screenshot.copy()
                        cv2.rectangle(
                            debug_screenshot,
                            (canvas_area["x"], canvas_area["y"]),
                            (
                                canvas_area["x"] + canvas_area["width"],
                                canvas_area["y"] + canvas_area["height"],
                            ),
                            (0, 255, 0),
                            2,
                        )
                        debug_filename = (
                            f"debug_screenshots/monitor_{i}_canvas_found.png"
                        )
                        cv2.imwrite(
                            debug_filename,
                            cv2.cvtColor(debug_screenshot, cv2.COLOR_RGB2BGR),
                        )
                        self.logger.info(
                            f"Saved canvas area screenshot: {debug_filename}"
                        )

                        return {
                            "capture_area": global_canvas_area,
                            "monitor": monitor,
                            "screenshot": screenshot,
                        }

                self.logger.warning("Canvas not found on any monitor")
                return None

        except Exception as e:
            self.logger.error("Error capturing screens", e)
            self.state = BotState.ERROR
            self.error_message = f"Screen capture error: {str(e)}"
            return None

    def save_settings(self) -> None:
        try:
            with open("trex_settings.json", "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            self.logger.info("Settings saved")
        except Exception as e:
            self.logger.error("Error saving settings", e)

    def load_settings(self) -> None:
        try:
            with open("trex_settings.json", "r", encoding="utf-8") as f:
                loaded_settings = json.load(f)
                self.settings.update(loaded_settings)
            self.logger.info("Settings loaded")
        except Exception as e:
            self.logger.error("Error loading settings", e)

    def save_stats(self) -> None:
        try:
            with open("trex_stats.json", "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            self.logger.info("Statistics saved")
        except Exception as e:
            self.logger.error("Error saving statistics", e)

    def load_stats(self) -> None:
        try:
            with open("trex_stats.json", "r", encoding="utf-8") as f:
                loaded_stats = json.load(f)
                self.stats.update(loaded_stats)
            self.logger.info("Statistics loaded")
        except Exception as e:
            self.logger.error("Error loading statistics", e)

    def reset_stats(self) -> None:
        try:
            self.stats = {
                "games_played": 0,
                "best_score": 0,
                "total_time": 0,
                "total_jumps": 0,
                "total_ducks": 0,
            }
            self.save_stats()
            self.logger.info("Statistics reset")
        except Exception as e:
            self.logger.error("Error resetting statistics", e)
