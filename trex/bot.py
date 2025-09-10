import mss
import numpy as np
import cv2
import time
import keyboard
import pyautogui
import threading
import json
import os
from datetime import datetime


class BotLogic:
    def __init__(self):
        self.is_running = False
        self.bot_thread = None
        self.start_time = None

        self.settings = {
            "detection_threshold": 0.95,
            "jump_delay": 0.1,
            "scan_delay": 0.01,
            "obstacle_distance": 150,
        }

        self.stats = {
            "games_played": 0,
            "best_score": 0,
            "total_time": 0,
            "total_jumps": 0,
        }

        self.update_status_callback = None
        self.update_stats_callback = None
        self.get_capture_area_callback = None
        self.change_control_button_callback = None

    def set_callbacks(
        self, update_status, update_stats, get_capture_area, change_control_button
    ):
        self.update_status_callback = update_status
        self.update_stats_callback = update_stats
        self.get_capture_area_callback = get_capture_area
        self.change_control_button_callback = change_control_button

    def start_bot(self):
        self.is_running = True
        self.change_control_button_callback("⏹ Стоп", "#cc2936")
        self.update_status_callback("Запущен", "red")
        self.bot_thread = threading.Thread(target=self.run_bot, daemon=True)
        self.bot_thread.start()

    def stop_bot(self):
        self.is_running = False
        self.change_control_button_callback("▶ Старт", "#0d7377")
        self.update_status_callback("Остановлен", "#90ee90")

    def run_bot(self):
        try:
            while self.is_running:
                if keyboard.is_pressed("space"):
                    break
                time.sleep(0.1)

            if not self.is_running:
                return

            self.start_time = time.time()
            self.update_status_callback("Играет", "yellow")

            jumps_count = 0

            while self.is_running:
                if keyboard.is_pressed("esc"):
                    break

                if self.detect_obstacle():
                    pyautogui.press("space")
                    jumps_count += 1
                    time.sleep(self.settings["jump_delay"])

                time.sleep(self.settings["scan_delay"])

            if self.start_time:
                game_time = time.time() - self.start_time
                score = int(game_time * 10)

                self.stats["games_played"] += 1
                self.stats["total_time"] += game_time
                self.stats["total_jumps"] += jumps_count

                if score > self.stats["best_score"]:
                    self.stats["best_score"] = score

                self.update_stats_callback()
                self.save_stats()

        except Exception as e:
            print(f"Ошибка: {e}")
        finally:
            if self.is_running:
                self.stop_bot()

    def detect_obstacle(self):
        try:
            capture_area = self.get_capture_area_callback()

            with mss.mss() as sct:
                screenshot = sct.grab(capture_area)
                img = np.array(screenshot)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                height, width = gray.shape
                if height < 50 or width < 50:
                    return False

                roi_y = int(height * 0.7)
                roi_x = int(width * 0.3)
                roi_width = int(width * 0.4)
                roi_height = int(height * 0.2)

                roi = gray[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

                _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)

                obstacle_pixels = cv2.countNonZero(thresh)
                total_pixels = roi_width * roi_height

                if total_pixels > 0:
                    density = obstacle_pixels / total_pixels
                    return density > 0.1

                return False

        except Exception as e:
            self.log_callback(f"Ошибка детекции: {e}")
            return False

    def save_settings(self):
        try:
            with open("trex_settings.json", "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            self.log_callback(f"Ошибка сохранения настроек: {e}")

    def load_settings(self):
        try:
            if os.path.exists("trex_settings.json"):
                with open("trex_settings.json", "r", encoding="utf-8") as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            self.log_callback(f"Ошибка загрузки настроек: {e}")

    def save_stats(self):
        try:
            with open("trex_stats.json", "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            self.log_callback(f"Ошибка сохранения статистики: {e}")

    def load_stats(self):
        try:
            if os.path.exists("trex_stats.json"):
                with open("trex_stats.json", "r", encoding="utf-8") as f:
                    self.stats.update(json.load(f))
        except Exception as e:
            self.log_callback(f"Ошибка загрузки статистики: {e}")

    def reset_stats(self):
        self.stats = {
            "games_played": 0,
            "best_score": 0,
            "total_time": 0,
            "total_jumps": 0,
        }
        self.save_stats()
