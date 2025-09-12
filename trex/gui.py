import tkinter as tk
from tkinter import messagebox
from bot import TRexBot, BotState
from debug import DebugWindow, Logger


class TRexGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TRex Bot")
        self.root.geometry("400x600")
        self.root.resizable(False, False)
        self.root.configure(bg="#2b2b2b")
        self.root.attributes("-topmost", True)

        self.status_label = None
        self.control_button = None
        self.stats_vars = {}

        self.logger = Logger()
        self.bot = TRexBot(self.logger)
        self.debug_window = DebugWindow(self.bot, self.logger)

        self.last_bot_state = BotState.READY
        self.debug_shown = False
        self.game_over_dialog_shown = False

        self.create_widgets()
        self.load_settings()
        self.update_stats()

        self.start_gui_updates()

    def create_widgets(self) -> None:
        main_frame = tk.Frame(self.root, bg="#2b2b2b", padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        title_label = tk.Label(
            main_frame,
            text="TRex Bot",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#2b2b2b",
        )
        title_label.pack(pady=(0, 20))

        self.status_label = tk.Label(
            main_frame,
            text="Готов к работе",
            font=("Arial", 10),
            fg="#90ee90",
            bg="#2b2b2b",
        )
        self.status_label.pack(pady=(0, 10))

        self.control_button = tk.Button(
            main_frame,
            text="▶ Старт",
            font=("Arial", 12, "bold"),
            bg="#0d7377",
            fg="white",
            relief="flat",
            width=15,
            height=2,
            command=self.toggle_bot,
        )
        self.control_button.pack(pady=(0, 20))

        info_frame = tk.LabelFrame(
            main_frame,
            text="Информация",
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#2b2b2b",
            relief="flat",
        )
        info_frame.pack(fill="x", pady=(0, 15))

        info_text = tk.Text(
            info_frame,
            height=4,
            bg="#3b3b3b",
            fg="white",
            font=("Arial", 8),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )

        settings_frame = tk.LabelFrame(
            main_frame,
            text="Настройки",
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#2b2b2b",
            relief="flat",
        )
        settings_frame.pack(fill="x", pady=(0, 15))

        settings = [
            ("Чувствительность", "obstacle_density", 0.01, 0.3),
            ("Задержка сканирования", "scan_delay", 0.001, 0.1),
            ("Задержка прыжка", "jump_delay", 0.01, 0.5),
            ("Порог уверенности", "confidence_threshold", 0.1, 0.99),
        ]

        for i, (label, key, min_val, max_val) in enumerate(settings):
            frame = tk.Frame(settings_frame, bg="#2b2b2b")
            frame.pack(fill="x", pady=2)

            tk.Label(
                frame,
                text=label,
                font=("Arial", 9),
                fg="white",
                bg="#2b2b2b",
                width=20,
                anchor="w",
            ).pack(side="left")

            var = tk.DoubleVar(value=self.bot.settings[key])
            setattr(self, f"{key}_var", var)

            scale = tk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                resolution=0.001 if key == "obstacle_density" else 0.01,
                orient="horizontal",
                variable=var,
                bg="#2b2b2b",
                fg="white",
                highlightthickness=0,
                length=150,
                command=lambda val, k=key: self.on_setting_change(k, float(val)),
            )
            scale.pack(side="right")

        stats_frame = tk.LabelFrame(
            main_frame,
            text="Статистика",
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#2b2b2b",
            relief="flat",
        )
        stats_frame.pack(fill="x", pady=(0, 15))

        self.stats_vars = {}
        stats_labels = [
            ("Игр сыграно", "games_played"),
            ("Лучший счёт", "best_score"),
            ("Общее время", "total_time"),
            ("Всего прыжков", "total_jumps"),
        ]

        for label, key in stats_labels:
            frame = tk.Frame(stats_frame, bg="#2b2b2b")
            frame.pack(fill="x", pady=2)

            tk.Label(
                frame,
                text=label,
                font=("Arial", 9),
                fg="white",
                bg="#2b2b2b",
                width=15,
                anchor="w",
            ).pack(side="left")

            var = tk.StringVar(value="0")
            self.stats_vars[key] = var

            tk.Label(
                frame,
                textvariable=var,
                font=("Arial", 9),
                fg="#90ee90",
                bg="#2b2b2b",
                width=10,
                anchor="e",
            ).pack(side="right")

        button_frame = tk.Frame(main_frame, bg="#2b2b2b")
        button_frame.pack(fill="x", pady=(10, 0))

        reset_stats_btn = tk.Button(
            button_frame,
            text="Сбросить статистику",
            font=("Arial", 10),
            bg="#555555",
            fg="white",
            relief="flat",
            command=self.reset_stats_with_confirmation,
        )
        reset_stats_btn.pack(side="left", padx=(0, 10))

        save_btn = tk.Button(
            button_frame,
            text="Сохранить настройки",
            font=("Arial", 10),
            bg="#555555",
            fg="white",
            relief="flat",
            command=self.bot.save_settings,
        )
        save_btn.pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_gui_updates(self):
        self.update_gui_state()
        self.root.after(100, self.start_gui_updates)

    def update_gui_state(self):
        try:
            current_state = self.bot.state

            if self.bot.is_running:
                self.control_button.config(text="⏹ Стоп", bg="#cc2936")
            else:
                self.control_button.config(text="▶ Старт", bg="#0d7377")

            status_text = self.bot.status_message
            if self.bot.error_message:
                status_text = self.bot.error_message

            status_color = self.get_status_color(current_state)
            self.status_label.config(text=status_text, fg=status_color)

            self.manage_debug_window()

            if current_state == BotState.GAME_OVER and not self.game_over_dialog_shown:
                self.handle_game_over_dialog()

            self.update_stats()

            self.last_bot_state = current_state

        except Exception as e:
            self.logger.error("Ошибка обновления GUI", e)

    def get_status_color(self, state: BotState) -> str:
        color_map = {
            BotState.READY: "#90ee90",
            BotState.SEARCHING_DINO: "orange",
            BotState.DINO_FOUND: "green",
            BotState.DINO_NOT_FOUND: "red",
            BotState.WAITING_FOR_GAME: "yellow",
            BotState.PLAYING: "yellow",
            BotState.GAME_OVER: "orange",
            BotState.STOPPED: "#90ee90",
            BotState.ERROR: "red",
        }
        return color_map.get(state, "white")

    def manage_debug_window(self):
        debug_info = self.bot.get_debug_info()

        should_show = (
            debug_info["should_show"] and debug_info["capture_area"] is not None
        )

        if should_show and not self.debug_shown:
            self.debug_window.show(debug_info["capture_area"])
            self.debug_shown = True
        elif not should_show and self.debug_shown:
            self.debug_window.hide()
            self.debug_shown = False

    def handle_game_over_dialog(self):
        try:
            self.game_over_dialog_shown = True

            game_stats = self.bot.game_stats
            score = game_stats.current_score
            game_time = game_stats.current_game_time
            jumps = game_stats.current_jumps

            message = f"""Игра завершена!
            
Результаты:
• Счёт: {score}
• Время игры: {game_time:.1f} сек
• Количество прыжков: {jumps}

Хотите начать новую игру?"""

            result = messagebox.askyesno(
                "T-Rex Bot - Игра завершена", message, icon="question"
            )

            if result:
                self.bot.restart_game()
            else:
                self.bot.stop_bot()

            self.game_over_dialog_shown = False

        except Exception as e:
            self.logger.error("Ошибка обработки диалога завершения игры", e)
            self.game_over_dialog_shown = False

    def reset_stats_with_confirmation(self):
        result = messagebox.askyesno(
            "Подтверждение",
            "Вы уверены, что хотите сбросить всю статистику?",
            icon="warning",
        )
        if result:
            self.bot.reset_stats()
            self.update_stats()

    def on_setting_change(self, key: str, value: float) -> None:
        self.bot.settings[key] = value

    def toggle_bot(self) -> None:
        if self.bot.is_running:
            self.bot.stop_bot()
        else:
            self.game_over_dialog_shown = False
            self.bot.start_bot()

    def update_stats(self) -> None:
        try:
            stats = self.bot.stats
            self.stats_vars["games_played"].set(str(int(stats["games_played"])))
            self.stats_vars["best_score"].set(str(int(stats["best_score"])))
            self.stats_vars["total_time"].set(f"{stats['total_time']:.1f}с")
            self.stats_vars["total_jumps"].set(str(int(stats["total_jumps"])))
        except Exception as e:
            self.logger.error("Ошибка обновления статистики в GUI", e)

    def load_settings(self) -> None:
        try:
            for key, value in self.bot.settings.items():
                var_name = f"{key}_var"
                if hasattr(self, var_name):
                    var = getattr(self, var_name)
                    var.set(value)
        except Exception as e:
            self.logger.error("Ошибка загрузки настроек в GUI", e)

    def on_closing(self):
        try:
            if self.bot.is_running:
                self.bot.stop_bot()

            if self.debug_shown:
                self.debug_window.hide()

            self.bot.save_settings()

            try:
                if self.root and self.root.winfo_exists():
                    self.root.destroy()
            except tk.TclError:
                pass

        except Exception as e:
            self.logger.error("Ошибка при закрытии приложения", e)

    def run(self) -> None:
        try:
            template_info = self.bot.object_detector.get_template_info()
            if template_info:
                self.logger.info(
                    f"Загружены шаблоны: {', '.join(template_info.keys())}"
                )
            else:
                messagebox.showwarning(
                    "Предупреждение",
                    "Не найдено ни одного шаблона в папке images/\n"
                    "Убедитесь, что файлы изображений находятся в папке images/",
                )

            self.root.mainloop()

        except Exception as e:
            self.logger.error("Критическая ошибка в GUI", e)
            messagebox.showerror("Ошибка", f"Критическая ошибка: {str(e)}")
        finally:
            try:
                if self.bot.is_running:
                    self.bot.stop_bot()
                self.bot.save_settings()
            except Exception as e:
                print(f"Ошибка при финальном сохранении: {e}")
