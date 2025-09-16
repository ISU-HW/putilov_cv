import tkinter as tk
from tkinter import messagebox, ttk
from bot import TRexBot, BotState
from logger import Logger


class TRexGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("T-Rex Bot v2.0")
        self.root.geometry("450x850")
        self.root.resizable(False, False)
        self.root.configure(bg="#2b2b2b")

        self.status_label = None
        self.control_button = None
        self.stats_vars = {}
        self.current_game_vars = {}

        self.logger = Logger()
        self.bot = TRexBot(self.logger)

        self.last_bot_state = BotState.READY
        self.game_over_dialog_shown = False

        self.create_widgets()
        self.load_settings()
        self.update_stats()

        self.start_gui_updates()

    def create_widgets(self) -> None:
        main_frame = tk.Frame(self.root, bg="#2b2b2b", padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = tk.Label(
            main_frame,
            text="T-Rex Bot v2.0",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#2b2b2b",
        )
        title_label.pack(pady=(0, 20))

        # Status
        status_frame = tk.LabelFrame(
            main_frame,
            text="Status",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#2b2b2b",
            relief="flat",
        )
        status_frame.pack(fill="x", pady=(0, 15))

        self.status_label = tk.Label(
            status_frame,
            text="Ready to start",
            font=("Arial", 11),
            fg="#90ee90",
            bg="#2b2b2b",
            pady=10,
        )
        self.status_label.pack()

        # Control button
        self.control_button = tk.Button(
            main_frame,
            text="▶ Start",
            font=("Arial", 14, "bold"),
            bg="#0d7377",
            fg="white",
            relief="flat",
            width=15,
            height=2,
            command=self.toggle_bot,
        )
        self.control_button.pack(pady=(0, 20))

        # Current game stats
        current_frame = tk.LabelFrame(
            main_frame,
            text="Current Game",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#2b2b2b",
            relief="flat",
        )
        current_frame.pack(fill="x", pady=(0, 15))

        current_stats = [
            ("Score", "current_score"),
            ("Time", "current_time"),
            ("Jumps", "current_jumps"),
            ("Ducks", "current_ducks"),
        ]

        self.current_game_vars = {}
        for label, key in current_stats:
            frame = tk.Frame(current_frame, bg="#2b2b2b")
            frame.pack(fill="x", pady=3)

            tk.Label(
                frame,
                text=f"{label}:",
                font=("Arial", 10),
                fg="white",
                bg="#2b2b2b",
                width=10,
                anchor="w",
            ).pack(side="left")

            var = tk.StringVar(value="0")
            self.current_game_vars[key] = var

            tk.Label(
                frame,
                textvariable=var,
                font=("Arial", 10, "bold"),
                fg="#ffdd44",
                bg="#2b2b2b",
                width=15,
                anchor="e",
            ).pack(side="right")

        settings_frame = tk.LabelFrame(
            main_frame,
            text="Settings",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#2b2b2b",
            relief="flat",
        )
        settings_frame.pack(fill="x", pady=(0, 15))

        settings = [
            ("Jump Sensitivity", "jump_sensitivity", 0.01, 0.2),
            ("Scan Delay (ms)", "scan_delay", 0.001, 0.1),
            ("Jump Delay (ms)", "jump_delay", 0.01, 0.5),
            ("Confidence", "confidence_threshold", 0.1, 0.99),
            ("Duck Duration (s)", "duck_duration", 0.1, 1.0),
        ]

        for label, key, min_val, max_val in settings:
            frame = tk.Frame(settings_frame, bg="#2b2b2b")
            frame.pack(fill="x", pady=3)

            tk.Label(
                frame,
                text=label,
                font=("Arial", 9),
                fg="white",
                bg="#2b2b2b",
                width=18,
                anchor="w",
            ).pack(side="left")

            var = tk.DoubleVar(value=self.bot.settings.get(key, min_val))
            setattr(self, f"{key}_var", var)

            scale = tk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                resolution=0.001 if "delay" in key or "sensitivity" in key else 0.01,
                orient="horizontal",
                variable=var,
                bg="#2b2b2b",
                fg="white",
                highlightthickness=0,
                length=180,
                command=lambda val, k=key: self.on_setting_change(k, float(val)),
            )
            scale.pack(side="right")

        stats_frame = tk.LabelFrame(
            main_frame,
            text="Overall Statistics",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#2b2b2b",
            relief="flat",
        )
        stats_frame.pack(fill="x", pady=(0, 15))

        stats_labels = [
            ("Games Played", "games_played"),
            ("Best Score", "best_score"),
            ("Total Time", "total_time"),
            ("Total Jumps", "total_jumps"),
            ("Total Ducks", "total_ducks"),
        ]

        self.stats_vars = {}
        for label, key in stats_labels:
            frame = tk.Frame(stats_frame, bg="#2b2b2b")
            frame.pack(fill="x", pady=3)

            tk.Label(
                frame,
                text=f"{label}:",
                font=("Arial", 10),
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
                font=("Arial", 10),
                fg="#90ee90",
                bg="#2b2b2b",
                width=12,
                anchor="e",
            ).pack(side="right")

        button_frame = tk.Frame(main_frame, bg="#2b2b2b")
        button_frame.pack(fill="x", pady=(15, 0))

        reset_stats_btn = tk.Button(
            button_frame,
            text="Reset Stats",
            font=("Arial", 10),
            bg="#cc2936",
            fg="white",
            relief="flat",
            command=self.reset_stats_with_confirmation,
        )
        reset_stats_btn.pack(side="left", padx=(0, 10))

        save_btn = tk.Button(
            button_frame,
            text="Save Settings",
            font=("Arial", 10),
            bg="#0d7377",
            fg="white",
            relief="flat",
            command=self.bot.save_settings,
        )
        save_btn.pack(side="right")

        # Info label
        info_label = tk.Label(
            main_frame,
            text="v2.0: Added duck action for pterodactyls",
            font=("Arial", 8),
            fg="#666666",
            bg="#2b2b2b",
        )
        info_label.pack(pady=(10, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_gui_updates(self):
        self.update_gui_state()
        self.root.after(100, self.start_gui_updates)

    def update_gui_state(self):
        try:
            current_state = self.bot.state

            if self.bot.is_running:
                self.control_button.config(text="⏹ Stop", bg="#cc2936")
            else:
                self.control_button.config(text="▶ Start", bg="#0d7377")

            status_text = self.bot.status_message
            if self.bot.error_message:
                status_text = self.bot.error_message

            status_color = self.get_status_color(current_state)
            self.status_label.config(text=status_text, fg=status_color)

            if current_state == BotState.GAME_OVER and not self.game_over_dialog_shown:
                self.handle_game_over_dialog()

            self.update_current_game_stats()
            self.update_stats()

            self.last_bot_state = current_state

        except Exception as e:
            self.logger.error("Error updating GUI", e)

    def get_status_color(self, state: BotState) -> str:
        color_map = {
            BotState.READY: "yellow",
            BotState.SEARCHING_CANVAS: "orange",
            BotState.CANVAS_FOUND: "green",
            BotState.CANVAS_NOT_FOUND: "red",
            BotState.WAITING_FOR_GAME: "yellow",
            BotState.PLAYING: "green",
            BotState.GAME_OVER: "orange",
            BotState.STOPPED: "red",
            BotState.ERROR: "red",
        }
        return color_map.get(state, "white")

    def handle_game_over_dialog(self):
        try:
            self.game_over_dialog_shown = True

            game_stats = self.bot.game_stats
            score = game_stats.current_score
            game_time = game_stats.current_game_time
            jumps = game_stats.current_jumps
            ducks = game_stats.current_ducks

            self.logger.info(
                f"Game Over! Score: {score}, Time: {game_time:.1f}s, Jumps: {jumps}, Ducks: {ducks}"
            )

            self.game_over_dialog_shown = False

        except Exception as e:
            self.logger.error("Error handling game over dialog", e)
            self.game_over_dialog_shown = False

    def update_current_game_stats(self):
        try:
            stats = self.bot.game_stats
            self.current_game_vars["current_score"].set(str(stats.current_score))
            self.current_game_vars["current_time"].set(
                f"{stats.current_game_time:.1f}s"
            )
            self.current_game_vars["current_jumps"].set(str(stats.current_jumps))
            self.current_game_vars["current_ducks"].set(str(stats.current_ducks))
        except Exception as e:
            self.logger.error("Error updating current game stats", e)

    def update_stats(self) -> None:
        try:
            stats = self.bot.stats
            self.stats_vars["games_played"].set(str(int(stats.get("games_played", 0))))
            self.stats_vars["best_score"].set(str(int(stats.get("best_score", 0))))
            self.stats_vars["total_time"].set(f"{stats.get('total_time', 0):.1f}s")
            self.stats_vars["total_jumps"].set(str(int(stats.get("total_jumps", 0))))
            self.stats_vars["total_ducks"].set(str(int(stats.get("total_ducks", 0))))
        except Exception as e:
            self.logger.error("Error updating statistics in GUI", e)

    def reset_stats_with_confirmation(self):
        result = messagebox.askyesno(
            "Confirm Reset",
            "Are you sure you want to reset all statistics?",
            icon="warning",
        )
        if result:
            self.bot.reset_stats()
            self.update_stats()

    def on_setting_change(self, key: str, value: float) -> None:
        self.bot.settings[key] = value

        if key == "confidence_threshold":
            self.bot.object_detector.set_confidence_threshold(value)

    def toggle_bot(self) -> None:
        if self.bot.is_running:
            self.bot.stop_bot()
        else:
            self.game_over_dialog_shown = False
            self.bot.start_bot()

    def load_settings(self) -> None:
        try:
            for key, value in self.bot.settings.items():
                var_name = f"{key}_var"
                if hasattr(self, var_name):
                    var = getattr(self, var_name)
                    var.set(value)
        except Exception as e:
            self.logger.error("Error loading settings in GUI", e)

    def on_closing(self):
        try:
            if self.bot.is_running:
                self.bot.stop_bot()

            self.bot.save_settings()

            try:
                if self.root and self.root.winfo_exists():
                    self.root.destroy()
            except tk.TclError:
                pass

        except Exception as e:
            self.logger.error("Error closing application", e)

    def run(self) -> None:
        try:
            template_info = self.bot.object_detector.get_template_info()
            if template_info:
                self.logger.info(f"Loaded templates: {', '.join(template_info.keys())}")
            else:
                messagebox.showwarning(
                    "Warning",
                    "No templates found in images/ directory.\n"
                    "Make sure template images are in the images/ folder.",
                )

            self.root.mainloop()

        except Exception as e:
            self.logger.error("Critical GUI error", e)
            messagebox.showerror("Error", f"Critical error: {str(e)}")
        finally:
            try:
                if self.bot.is_running:
                    self.bot.stop_bot()
                self.bot.save_settings()
            except Exception as e:
                print(f"Error during final cleanup: {e}")
