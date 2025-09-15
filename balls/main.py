import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import time


class ColorSequenceGame:
    def __init__(self):
        self.colors = {
            "red": {
                "hsv_lower": np.array([0, 100, 100]),
                "hsv_upper": np.array([10, 255, 255]),
                "bgr": (0, 0, 255),
                "name": "red",
            },
            "green": {
                "hsv_lower": np.array([40, 100, 100]),
                "hsv_upper": np.array([80, 255, 255]),
                "bgr": (0, 255, 0),
                "name": "green",
            },
            "blue": {
                "hsv_lower": np.array([100, 100, 100]),
                "hsv_upper": np.array([130, 255, 255]),
                "bgr": (255, 0, 0),
                "name": "blue",
            },
            "yellow": {
                "hsv_lower": np.array([20, 100, 100]),
                "hsv_upper": np.array([40, 255, 255]),
                "bgr": (0, 255, 255),
                "name": "yellow",
            },
        }

        self.frame_width = 640
        self.frame_height = 480
        self.cap = None
        self.secret_sequence = []
        self.detected_colors = []
        self.game_mode = "3balls"
        self.show_result = False
        self.result_time = 0

    def select_mode(self):
        print("\nSelect mode:")
        print("1. Test mode (analyze 3balls.png and 4balls.png)")
        print("2. Game mode (camera + guessing game)")

        while True:
            choice = input("Choose mode (1 or 2): ").strip()
            if choice == "1":
                return "test"
            elif choice == "2":
                return "game"
            else:
                print("Invalid choice. Enter 1 or 2")

    def detect_color_in_region(self, frame, region):
        x, y, w, h = region
        roi = frame[y : y + h, x : x + w]

        if roi.size == 0:
            return None

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        best_color = None
        max_pixels = 0

        for color_name, color_info in self.colors.items():
            mask = cv2.inRange(
                hsv_roi, color_info["hsv_lower"], color_info["hsv_upper"]
            )
            color_pixels = cv2.countNonZero(mask)

            if color_pixels > max_pixels and color_pixels > 100:
                max_pixels = color_pixels
                best_color = color_name

        return best_color

    def get_detection_regions(self):
        if self.game_mode == "3balls":
            region_width = 80
            region_height = 80
            y_center = self.frame_height // 2 - region_height // 2

            regions = []
            for i in range(3):
                x = (self.frame_width // 4) * (i + 1) - region_width // 2
                regions.append((x, y_center, region_width, region_height))
            return regions
        else:
            region_width = 70
            region_height = 70

            regions = []
            for row in range(2):
                for col in range(2):
                    x = (self.frame_width // 3) * (col + 1) - region_width // 2
                    y = (self.frame_height // 3) * (row + 1) - region_height // 2
                    regions.append((x, y, region_width, region_height))
            return regions

    def generate_sequence(self):
        length = 3 if self.game_mode == "3balls" else 4
        color_names = list(self.colors.keys())
        self.secret_sequence = [random.choice(color_names) for _ in range(length)]
        print(f"New sequence generated ({length} colors)")

    def draw_detection_areas(self, frame):
        regions = self.get_detection_regions()

        for i, (x, y, w, h) in enumerate(regions):
            if self.show_result and i < len(self.detected_colors):

                if (
                    i < len(self.secret_sequence)
                    and self.detected_colors[i] == self.secret_sequence[i]
                ):
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            else:

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            cv2.putText(
                frame,
                str(i + 1),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def draw_secret_sequence(self, frame):
        if not self.secret_sequence or not self.show_result:
            return

        cv2.putText(
            frame,
            "Target sequence:",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if self.game_mode == "3balls":
            for i, color_name in enumerate(self.secret_sequence):
                x = 50 + i * 60
                y = 50
                color_bgr = self.colors[color_name]["bgr"]
                cv2.circle(frame, (x, y), 20, color_bgr, -1)
                cv2.circle(frame, (x, y), 20, (255, 255, 255), 2)
        else:
            for i, color_name in enumerate(self.secret_sequence):
                row = i // 2
                col = i % 2
                x = 50 + col * 60
                y = 50 + row * 50
                color_bgr = self.colors[color_name]["bgr"]
                cv2.circle(frame, (x, y), 20, color_bgr, -1)
                cv2.circle(frame, (x, y), 20, (255, 255, 255), 2)

    def draw_instructions(self, frame):
        instructions = [
            "SPACE - guess/new game",
            "1 - 3 balls mode",
            "2 - 4 balls mode",
            "ESC - exit",
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (frame.shape[1] - 200, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def process_frame(self, frame):
        regions = self.get_detection_regions()
        detected = []

        for region in regions:
            color = self.detect_color_in_region(frame, region)
            detected.append(color)

        self.detected_colors = detected
        return frame

    def check_guess(self):
        if not self.secret_sequence:
            return False

        if len(self.detected_colors) != len(self.secret_sequence):
            return False

        correct = True
        for i in range(len(self.secret_sequence)):
            if self.detected_colors[i] != self.secret_sequence[i]:
                correct = False
                break

        return correct

    def run_game_mode(self):
        print("\nGame mode started")
        print("Controls:")
        print("SPACE - make guess / start new game")
        print("1 - switch to 3 balls mode")
        print("2 - switch to 4 balls mode")
        print("ESC - exit")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        self.generate_sequence()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break

                frame = cv2.flip(frame, 1)
                frame = self.process_frame(frame)

                if self.show_result and time.time() - self.result_time > 3:
                    self.show_result = False

                self.draw_detection_areas(frame)
                self.draw_secret_sequence(frame)
                self.draw_instructions(frame)

                cv2.imshow("Color Sequence Game", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break
                elif key == ord(" "):
                    if self.check_guess():
                        print("CORRECT!")
                    else:
                        print("INCORRECT!")

                    self.show_result = True
                    self.result_time = time.time()

                    self.generate_sequence()

                elif key == ord("1"):
                    self.game_mode = "3balls"
                    self.generate_sequence()
                    print("Switched to 3 balls mode")

                elif key == ord("2"):
                    self.game_mode = "4balls"
                    self.generate_sequence()
                    print("Switched to 4 balls mode")

        except KeyboardInterrupt:
            print("\nGame stopped by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def analyze_image(self, filename, mode):
        try:
            image = cv2.imread(filename)
            if image is None:
                print(f"Error: Could not load {filename}")
                return None, None

            image = cv2.resize(image, (self.frame_width, self.frame_height))

            old_mode = self.game_mode
            self.game_mode = mode
            regions = self.get_detection_regions()
            self.game_mode = old_mode

            detected_colors = []
            for region in regions:
                color = self.detect_color_in_region(image, region)
                detected_colors.append(color if color else "none")

            return image, detected_colors

        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            return None, None

    def run_test_mode(self):
        print("\nTest mode - analyzing images...")

        plt.rcParams["font.size"] = 12
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Color Detection Analysis", fontsize=16)

        image_3, colors_3 = self.analyze_image("3balls.png", "3balls")
        if image_3 is not None:
            image_3_rgb = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)

            axes[0, 0].imshow(image_3_rgb)
            axes[0, 0].set_title("3balls.png - Original")
            axes[0, 0].axis("off")

            old_mode = self.game_mode
            self.game_mode = "3balls"
            regions_3 = self.get_detection_regions()
            self.game_mode = old_mode

            for i, (x, y, w, h) in enumerate(regions_3):
                rect = Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor="white", facecolor="none"
                )
                axes[0, 0].add_patch(rect)
                axes[0, 0].text(
                    x + w // 2,
                    y - 10,
                    str(i + 1),
                    ha="center",
                    va="bottom",
                    color="white",
                    fontweight="bold",
                    fontsize=14,
                )

            axes[0, 1].bar(
                range(len(colors_3)),
                [1] * len(colors_3),
                color=[c if c != "none" else "gray" for c in colors_3],
            )
            axes[0, 1].set_title("Detected Colors (3 balls)")
            axes[0, 1].set_xticks(range(len(colors_3)))
            axes[0, 1].set_xticklabels([f"{i+1}: {c}" for i, c in enumerate(colors_3)])
            axes[0, 1].set_ylabel("Detection")
            axes[0, 1].set_ylim(0, 1.2)

        image_4, colors_4 = self.analyze_image("4balls.png", "4balls")
        if image_4 is not None:
            image_4_rgb = cv2.cvtColor(image_4, cv2.COLOR_BGR2RGB)

            axes[1, 0].imshow(image_4_rgb)
            axes[1, 0].set_title("4balls.png - Original")
            axes[1, 0].axis("off")

            old_mode = self.game_mode
            self.game_mode = "4balls"
            regions_4 = self.get_detection_regions()
            self.game_mode = old_mode

            for i, (x, y, w, h) in enumerate(regions_4):
                rect = Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor="white", facecolor="none"
                )
                axes[1, 0].add_patch(rect)
                axes[1, 0].text(
                    x + w // 2,
                    y - 10,
                    str(i + 1),
                    ha="center",
                    va="bottom",
                    color="white",
                    fontweight="bold",
                    fontsize=14,
                )

            axes[1, 1].bar(
                range(len(colors_4)),
                [1] * len(colors_4),
                color=[c if c != "none" else "gray" for c in colors_4],
            )
            axes[1, 1].set_title("Detected Colors (4 balls)")
            axes[1, 1].set_xticks(range(len(colors_4)))
            axes[1, 1].set_xticklabels([f"{i+1}: {c}" for i, c in enumerate(colors_4)])
            axes[1, 1].set_ylabel("Detection")
            axes[1, 1].set_ylim(0, 1.2)

        plt.tight_layout()
        plt.show()

        if colors_3:
            print(f"3balls.png: {colors_3}")
        if colors_4:
            print(f"4balls.png: {colors_4}")

    def run(self):
        print("Color Sequence Game")

        mode = self.select_mode()

        if mode == "test":
            self.run_test_mode()
        else:
            self.run_game_mode()


def main():
    game = ColorSequenceGame()
    game.run()


if __name__ == "__main__":
    main()
