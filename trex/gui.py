import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from datetime import datetime
import os
from bot import BotLogic


class TRexGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("T-Rex Bot")
        self.root.geometry("620x350+100+100")
        self.root.configure(bg="#2d2d2d")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        self.capture_width = 600
        self.capture_height = 150

        self.bot = BotLogic()

        self.create_widgets()

        self.bot.set_callbacks(
            update_status=lambda text, fg: self.status_label.config(text=text, fg=fg),
            update_stats=self.update_stats_display,
            get_capture_area=self.get_capture_area,
            change_control_button=lambda text, bg: self.control_button.config(
                text=text, bg=bg
            ),
        )

        self.bot.load_stats()
        self.update_stats_display()
        self.bot.load_settings()
        self.update_settings_display()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        control_frame = tk.Frame(self.root, bg="#2d2d2d", height=140)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        control_frame.pack_propagate(False)

        top_row = tk.Frame(control_frame, bg="#2d2d2d")
        top_row.pack(fill=tk.X, pady=2)

        game_button = tk.Button(
            top_row,
            text="Игра",
            command=self.open_game,
            bg="#4a4a4a",
            fg="white",
            relief="flat",
            padx=10,
        )
        game_button.pack(side=tk.LEFT, padx=2)

        self.control_button = tk.Button(
            top_row,
            text="▶ Старт",
            command=self.toggle_bot,
            bg="#0d7377",
            fg="white",
            relief="flat",
            padx=10,
        )
        self.control_button.pack(side=tk.LEFT, padx=2)

        self.status_label = tk.Label(
            top_row, text="Готов", bg="#2d2d2d", fg="#90ee90", font=("Arial", 9)
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        reset_button = tk.Button(
            top_row,
            text="↻",
            command=self.reset_stats,
            bg="#4a4a4a",
            fg="white",
            relief="flat",
            width=3,
        )
        reset_button.pack(side=tk.RIGHT, padx=2)

        stats_row = tk.Frame(control_frame, bg="#2d2d2d")
        stats_row.pack(fill=tk.X, pady=2)

        self.create_stats_display(stats_row)

        settings_row = tk.Frame(control_frame, bg="#2d2d2d")
        settings_row.pack(fill=tk.X, pady=2)

        self.create_settings_widgets(settings_row)

        self.capture_frame = tk.Frame(self.root, bg="red", relief="solid", bd=3)
        self.capture_frame.pack(padx=5, pady=(0, 5))

        self.capture_canvas = tk.Canvas(
            self.capture_frame,
            bg="#00ff00",
            highlightthickness=0,
            width=self.capture_width,
            height=self.capture_height,
        )
        self.capture_canvas.pack(padx=2, pady=2)

        info_frame = tk.Frame(self.root, bg="#2d2d2d", height=25)
        info_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        info_frame.pack_propagate(False)

        self.info_label = tk.Label(
            info_frame,
            text=f"Область захвата: {self.capture_width}x{self.capture_height} пикселей",
            bg="#2d2d2d",
            fg="white",
            font=("Arial", 8),
        )
        self.info_label.pack(pady=3)

    def create_stats_display(self, parent):
        self.stats_vars = {}

        stats_info = [
            ("games_played", "Игр:"),
            ("best_score", "Рекорд:"),
            ("total_time", "Время:"),
            ("total_jumps", "Прыжков:"),
        ]

        for i, (stat, label) in enumerate(stats_info):
            stat_frame = tk.Frame(parent, bg="#2d2d2d")
            stat_frame.pack(side=tk.LEFT, padx=10)

            tk.Label(
                stat_frame, text=label, bg="#2d2d2d", fg="white", font=("Arial", 8)
            ).pack()

            var = tk.StringVar(value=str(self.bot.stats[stat]))
            self.stats_vars[stat] = var
            tk.Label(
                stat_frame,
                textvariable=var,
                bg="#2d2d2d",
                fg="yellow",
                font=("Arial", 9, "bold"),
            ).pack()

    def create_settings_widgets(self, parent):
        self.settings_vars = {}

        settings_info = [
            ("jump_delay", "Задержка прыжка:"),
            ("scan_delay", "Сканирование:"),
            ("detection_threshold", "Порог детекции:"),
        ]

        for i, (setting, label) in enumerate(settings_info):
            frame = tk.Frame(parent, bg="#2d2d2d")
            frame.pack(side=tk.LEFT, padx=5)

            tk.Label(
                frame, text=label, bg="#2d2d2d", fg="white", font=("Arial", 8)
            ).pack()

            var = tk.StringVar(value=str(self.bot.settings[setting]))
            self.settings_vars[setting] = var
            entry = tk.Entry(
                frame,
                textvariable=var,
                width=8,
                bg="#4a4a4a",
                fg="white",
                relief="flat",
                font=("Arial", 8),
            )
            entry.pack()

            entry.bind("<FocusOut>", lambda e, s=setting: self.on_setting_change(s))
            entry.bind("<Return>", lambda e, s=setting: self.on_setting_change(s))

    def on_setting_change(self, setting_name):
        if self.bot.is_running:
            messagebox.showwarning(
                "Предупреждение", "Нельзя изменять настройки во время работы бота!"
            )
            self.settings_vars[setting_name].set(str(self.bot.settings[setting_name]))
            return

        try:
            new_value = float(self.settings_vars[setting_name].get())

            if new_value < 0:
                raise ValueError("Значение не может быть отрицательным")

            if setting_name == "detection_threshold" and (
                new_value < 0 or new_value > 1
            ):
                raise ValueError("Порог детекции должен быть от 0 до 1")

            self.bot.settings[setting_name] = new_value
            self.bot.save_settings()

        except ValueError as e:
            messagebox.showerror("Ошибка", f"Неверное значение: {e}")
            self.settings_vars[setting_name].set(str(self.bot.settings[setting_name]))

    def toggle_bot(self):
        if self.bot.is_running:
            self.bot.stop_bot()
        else:
            self.bot.start_bot()

    def get_capture_area(self):
        frame_x = self.capture_frame.winfo_rootx()
        frame_y = self.capture_frame.winfo_rooty()

        return {
            "top": frame_y,
            "left": frame_x,
            "width": self.capture_width,
            "height": self.capture_height,
        }

    def open_game(self):
        try:
            if os.path.exists("chromedino.html"):
                game_path = os.path.abspath("chromedino.html")
                webbrowser.open(f"file://{game_path}")
            else:
                messagebox.showerror("Ошибка", "Файл chromedino.html не найден")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть игру: {e}")

    def update_stats_display(self):
        self.stats_vars["games_played"].set(str(self.bot.stats["games_played"]))
        self.stats_vars["best_score"].set(str(self.bot.stats["best_score"]))
        self.stats_vars["total_time"].set(f"{self.bot.stats['total_time']:.0f}с")
        self.stats_vars["total_jumps"].set(str(self.bot.stats["total_jumps"]))

    def update_settings_display(self):
        for setting, var in self.settings_vars.items():
            var.set(str(self.bot.settings[setting]))

    def reset_stats(self):
        if messagebox.askyesno("Подтверждение", "Сбросить статистику?"):
            self.bot.reset_stats()
            self.update_stats_display()

    def on_closing(self):
        if self.bot.is_running:
            self.bot.stop_bot()
        self.bot.save_stats()
        self.root.destroy()
