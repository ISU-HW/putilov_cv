import tkinter as tk
from tkinter import Canvas, messagebox
import mss
import numpy as np
import cv2
import threading
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Callable


class Logger:
    def __init__(self):
        self.logs: List[Dict] = []
        self.max_logs = 1000
        self.callbacks: List[Callable] = []

    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)

    def _format_traceback(self, exception: Exception) -> Optional[str]:
        try:
            if hasattr(exception, "__traceback__") and exception.__traceback__:
                return "".join(
                    traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                )
            else:
                import inspect

                current_frame = inspect.currentframe()
                caller_frame = (
                    current_frame.f_back.f_back
                    if current_frame and current_frame.f_back
                    else None
                )
                if caller_frame:
                    return f"Exception in {caller_frame.f_code.co_filename}:{caller_frame.f_lineno} in {caller_frame.f_code.co_name}()\n{type(exception).__name__}: {str(exception)}"
        except:
            pass
        return f"{type(exception).__name__}: {str(exception)}"

    def log(
        self,
        level: str,
        message: str,
        exception: Optional[Exception] = None,
        include_stack: bool = False,
    ):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        formatted_traceback = None
        if exception:
            formatted_traceback = self._format_traceback(exception)
        elif include_stack:
            formatted_traceback = "".join(traceback.format_stack()[:-1])

        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "exception": str(exception) if exception else None,
            "traceback": formatted_traceback,
            "has_details": bool(exception or include_stack),
        }

        self.logs.append(log_entry)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

        print(f"[{timestamp}] {level}: {message}")
        if exception:
            print(f"Exception: {exception}")
        if formatted_traceback:
            print(f"Traceback:\n{formatted_traceback}")

        for callback in self.callbacks[:]:
            try:
                callback(log_entry)
            except Exception as e:
                print(f"Error in log callback: {e}")

    def info(self, message: str, include_stack: bool = False):
        self.log("INFO", message, include_stack=include_stack)

    def warning(self, message: str, include_stack: bool = False):
        self.log("WARNING", message, include_stack=include_stack)

    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        include_stack: bool = False,
    ):
        self.log("ERROR", message, exception, include_stack)

    def debug(self, message: str, include_stack: bool = False):
        self.log("DEBUG", message, include_stack=include_stack)


class DebugWindow:
    def __init__(self, bot_logic, logger: Logger):
        self.bot = bot_logic
        self.logger = logger
        self.window: Optional[tk.Toplevel] = None
        self.canvas: Optional[Canvas] = None
        self.is_visible: bool = False
        self.capture_area: Optional[Dict] = None
        self.running: bool = False

    def show(self, capture_area: Dict) -> None:
        try:
            self.capture_area = capture_area
            if not self.window:
                self.create_window()
            self.is_visible = self.running = True
            self.window.geometry(
                f"{capture_area['width']}x{capture_area['height']}+{capture_area['left']}+{capture_area['top']}"
            )
            self.window.deiconify()
            self.window.lift()
            self.window.attributes("-topmost", True)
            threading.Thread(target=self.update_loop, daemon=True).start()
        except Exception as e:
            self.logger.error("Ошибка показа debug окна", e)

    def create_window(self) -> None:
        try:
            self.window = tk.Toplevel()
            self.window.title("T-Rex Debug")
            self.window.configure(bg="white")
            self.window.overrideredirect(True)
            self.window.attributes("-topmost", True)
            self.window.attributes("-alpha", 0.5)

            self.canvas = Canvas(
                self.window,
                bg="white",
                highlightthickness=0,
                width=600,
                height=150,
            )
            self.canvas.pack(fill="both", expand=True)

            self.window.protocol("WM_DELETE_WINDOW", self.hide)
        except Exception as e:
            self.logger.error("Ошибка создания debug окна", e)

    def hide(self) -> None:
        try:
            self.is_visible = self.running = False
            if self.window:
                try:
                    self.window.withdraw()
                except tk.TclError:
                    pass
        except Exception as e:
            self.logger.error("Ошибка скрытия debug окна", e)

    def update_loop(self) -> None:
        try:
            with mss.mss() as sct:
                frame_count = 0
                while self.running and self.window and self.capture_area:
                    try:
                        screenshot = np.array(sct.grab(self.capture_area))

                        detected_objects = getattr(
                            self.bot, "last_detected_objects", []
                        )
                        trajectory = getattr(self.bot, "last_trajectory", [])
                        dino_info = getattr(self.bot, "last_dino_info", None)

                        self.draw_debug_info(
                            screenshot, detected_objects, trajectory, dino_info
                        )

                        frame_count += 1
                        if frame_count % 20 == 0:
                            try:
                                self.window.attributes("-topmost", True)
                            except tk.TclError:
                                pass

                        time.sleep(0.05)
                    except Exception as e:
                        self.logger.error("Ошибка обновления debug окна", e)
                        time.sleep(1)
        except Exception as e:
            self.logger.error("Ошибка в debug цикле", e)

    def draw_debug_info(
        self,
        screenshot: np.ndarray,
        detected_objects=None,
        trajectory=None,
        dino_info=None,
    ) -> None:
        """Минимальное отображение только обводок объектов"""
        if not self.canvas:
            return

        try:
            if not self.canvas.winfo_exists():
                return

            self.canvas.delete("all")

            height, width = screenshot.shape[:2]

            if detected_objects:
                for obj in detected_objects:
                    bbox = obj["bbox"]
                    obj_type = obj["type"]

                    if obj.get("is_threat", False):
                        color = "red"
                        width_line = 3
                    else:
                        color = "blue"
                        width_line = 2

                    self.canvas.create_rectangle(
                        bbox["x"],
                        bbox["y"],
                        bbox["x"] + bbox["width"],
                        bbox["y"] + bbox["height"],
                        outline=color,
                        width=width_line,
                        fill="",
                    )

            if trajectory and len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    x1, y1 = trajectory[i - 1]
                    x2, y2 = trajectory[i]
                    self.canvas.create_line(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill="green",
                        width=2,
                        dash=(5, 3),
                    )

            if (
                dino_info
                and isinstance(dino_info, dict)
                and "x" in dino_info
                and "y" in dino_info
            ):

                self.canvas.create_oval(
                    dino_info["x"] - 5,
                    dino_info["y"] - 5,
                    dino_info["x"] + 5,
                    dino_info["y"] + 5,
                    outline="yellow",
                    width=2,
                    fill="",
                )

        except tk.TclError:
            self.running = False
        except Exception as e:
            self.logger.error("Ошибка отрисовки debug информации", e)
