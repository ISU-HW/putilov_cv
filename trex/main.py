import keyboard
import mss
import time
import matplotlib.pyplot as plt
import pyautogui
import numpy as np
import cv2


## константы
start_distance_trex_detect_obstacles = 180
start_substraction = 0

## пути
trex_path = "images/trex.png"
grey_pixel_path = "images/grey_pixel.png"
trex_in_jump_path = "images/trex_in_jump.png"
gameover_path = "images/gameover.png"

obstacles = {
    "triple_cactus": ["images/triple_cactus.png", 0],
    "double_cactus": ["images/double_cactus.png", 0],
    "cactus": ["images/cactus.png", 0],
    "mini_triple_cactus": ["images/triple_mini_cactus.png", 1],
    "mini_double_cactus": ["images/double_mini_cactus.png", 1],
    "mini_cactus": ["images/mini_cactus.png", 1],
    "pterodactyl": ["images/pterodactyl.png", 0, False],
}


def find_obstacle_coordinates(
    obstacle_path,
    monitor=None,
    is_pterodactyl=False,
    trex_coordinates=None,
    obstacle=None,
):

    obstacle_img = cv2.imread(obstacle_path, cv2.IMREAD_GRAYSCALE)
    if obstacle_img is not None:
        # Захват скриншота с помощью mss
        with mss.mss() as sct:
            # Получение информации о мониторе
            if monitor is None:
                monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)

            if monitor["height"] == 1080:
                monitor["left"] = 0
                monitor["height"] = 0

            # Преобразование скриншота в OpenCV формат
            screen_shot = np.array(screenshot)
            screen_shot = cv2.cvtColor(screen_shot, cv2.COLOR_BGR2GRAY)

            # Поиск объекта на скриншоте
            result = cv2.matchTemplate(screen_shot, obstacle_img, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            threshold = 0.99

            if max_val >= threshold:
                # Координаты объекта на скриншоте
                start_x = max_loc[0] + monitor["left"]
                start_y = max_loc[1] + (monitor["top"] - monitor["height"])
                end_x = (start_x + obstacle_img.shape[1]) + monitor["left"]
                end_y = (start_y + obstacle_img.shape[0]) + (
                    monitor["top"] - monitor["height"]
                )
                width = obstacle_img.shape[1]
                height = obstacle_img.shape[0]

                if is_pterodactyl:
                    print("eto pterodactyl!")
                    if abs(max_loc[1] - trex_coordinates["y1"]) > 270:
                        obstacles[obstacle][2] = False
                    else:
                        obstacles[obstacle][2] = True

                return {
                    "x1": start_x,
                    "x2": end_x,
                    "y1": start_y,
                    "y2": end_y,
                    "height": height,
                    "width": width,
                    "path": obstacle_path,
                }

            else:
                return None


def if_trex_in_jump(trex_top):

    trex_coordinates = find_obstacle_coordinates(trex_path)
    current_trex_top = trex_coordinates["y2"]

    return not (trex_top == current_trex_top)


def calculate_time_in_quick_jump(trex_top):

    pyautogui.keyDown("space")
    pyautogui.keyUp("space")
    time1 = time.time()
    time.sleep(0.1)

    while if_trex_in_jump(trex_top):
        pass

    time2 = time.time()

    trex_in_jump = time2 - time1

    return trex_in_jump


def calculate_time_in_default_jump(trex_top):

    keyboard.press("space")
    time1 = time.time()
    time.sleep(0.1)

    while if_trex_in_jump(trex_top):
        pass

    time2 = time.time()

    trex_in_jump = time2 - time1

    return trex_in_jump


def calculate_speed(
    obstacle_coordinates, trex_coordinates, distance_trex_detect_obstacles
):

    left = int(trex_coordinates["x1"] - obstacle_coordinates["width"])
    top = int(trex_coordinates["y2"] - trex_coordinates["height"] * 2)
    width = int(
        distance_trex_detect_obstacles
        + trex_coordinates["width"]
        + obstacle_coordinates["width"]
    )
    height = int(trex_coordinates["height"] * 4)
    current_monitor = {"top": top, "left": left, "width": width, "height": height}

    obstacle_path = obstacle_coordinates["path"]

    current_obstacle_start1 = find_obstacle_coordinates(obstacle_path, current_monitor)[
        "x1"
    ]
    time1 = time.time()

    time.sleep(0.05)

    current_obstacle_coordinates = find_obstacle_coordinates(
        obstacle_path, current_monitor
    )
    current_obstacle_start2 = current_obstacle_coordinates["x1"]
    time2 = time.time()

    current_obstacle_coordinates["width"] * 1.1

    if obstacle_path == "images/pterodactyl.png":
        current_obstacle_coordinates["width"] = trex_coordinates["width"] * 1.3

    elif obstacle_path == "images/mini_cactus.png":
        obstacle_coordinates["width"] = 0

    elif obstacle_path == "images/cactus.png":
        current_obstacle_coordinates["width"] *= 1.5

    elif obstacle_path == "images/triple_cactus.png":
        current_obstacle_coordinates["width"] *= 5

    elif obstacle_path == "images/triple_mini_cactus.png":
        current_obstacle_coordinates["width"] *= 1.42

    distance = current_obstacle_start2 - trex_coordinates["x2"]

    speed = (current_obstacle_start1 - current_obstacle_start2) / (time2 - time1)

    return speed, distance, current_obstacle_coordinates


def calculate_wait_time_before_jump(
    obstacle_coordinates, trex_coordinates, time_in_jump, distance_trex_detect_obstacles
):

    speed, distance, obstacle_coordinates = calculate_speed(
        obstacle_coordinates, trex_coordinates, distance_trex_detect_obstacles
    )
    print(f"speed is {speed}")
    distance_before_jump = speed * time_in_jump
    wait_time_before_jump = 0

    obstalce_length = obstacle_coordinates["width"]

    wait_time_before_jump = (
        distance + trex_coordinates["width"] + obstalce_length - distance_before_jump
    ) / speed

    return wait_time_before_jump


def calculate_wait_time_after_jump(
    obstacle_coordinates, trex_coordinates, time_in_jump, distance_trex_detect_obstacles
):

    speed, distance, obstacle_coordinates = calculate_speed(
        obstacle_coordinates, trex_coordinates, distance_trex_detect_obstacles
    )

    obstalce_length = obstacle_coordinates["width"]

    wait_time_after_jump = (distance + obstalce_length) / speed

    return wait_time_after_jump


def jump(
    obstacle_coordinates,
    trex_coordinates,
    time_in_jump,
    obstacle,
    distance_trex_detect_obstacles,
    substraction,
    is_need_quick_jump,
):

    if obstacle_coordinates["x1"] - trex_coordinates["x2"] <= 100:
        wait_time_before_jump = 0
    else:
        wait_time_before_jump = calculate_wait_time_before_jump(
            obstacle_coordinates,
            trex_coordinates,
            time_in_jump,
            distance_trex_detect_obstacles,
        )

    button = "space"

    if obstacle == "pterodactyl":
        if obstacles[obstacle][2] == False:
            button = "down"

    if wait_time_before_jump > 0:
        time.sleep(wait_time_before_jump)

    if is_need_quick_jump:
        pyautogui.keyDown(button)
        if button == "space":
            pyautogui.keyUp(button)
    else:
        keyboard.press(button)

    try:
        wait_time_after_jump = calculate_wait_time_after_jump(
            obstacle_coordinates,
            trex_coordinates,
            time_in_jump,
            distance_trex_detect_obstacles,
        )
    except:
        wait_time_after_jump = 0
        print(wait_time_after_jump)

    substraction = 0
    wait_time_after_jump -= substraction

    if wait_time_after_jump > 0:
        time.sleep(wait_time_after_jump)

    pyautogui.keyDown("down")

    time.sleep(0.1)

    pyautogui.keyUp("down")


def play_trex():

    trex_coordinates = find_obstacle_coordinates(trex_path)

    print("trex is ready!")

    time_in_quick_jump = calculate_time_in_quick_jump(trex_coordinates["y2"])
    print(f"calculated time of quick jump is {time_in_quick_jump}")

    time_in_default_jump = calculate_time_in_default_jump(trex_coordinates["y2"])
    print(f"calculated time of default jump is {time_in_default_jump}")

    while True:

        if keyboard.is_pressed("ESC"):
            keyboard.release("down")
            return

        trex_coordinates = find_obstacle_coordinates(trex_path)

        for obstacle in obstacles:

            is_need_quick_jump = False
            time_in_jump = time_in_default_jump
            if obstacle == "mini_cactus":
                time_in_jump = time_in_quick_jump
                is_need_quick_jump = True

            obstacle_path = obstacles[obstacle][0]

            is_pterodactyl = False
            if obstacle == "pterodactyl":
                is_pterodactyl = True

            current_time = time.time()
            duration = current_time - start_time

            distance_trex_detect_obstacles = (
                start_distance_trex_detect_obstacles + duration * 2
            )
            try:
                left = int(trex_coordinates["x2"])
                top = int(trex_coordinates["y2"] - trex_coordinates["height"] * 2)
            except TypeError:
                continue

            width = int(distance_trex_detect_obstacles)
            height = int(trex_coordinates["height"] * 4)
            current_monitor = {
                "top": top,
                "left": left,
                "width": width,
                "height": height,
            }

            obstacle_coordinates = find_obstacle_coordinates(
                obstacle_path,
                current_monitor,
                is_pterodactyl,
                trex_coordinates,
                obstacle,
            )

            if obstacle_coordinates:

                print(f"trex found {obstacle}")

                current_time = time.time()
                duration = current_time - start_time

                substraction = start_substraction + (duration / 1600)

                jump(
                    obstacle_coordinates,
                    trex_coordinates,
                    time_in_jump,
                    obstacle,
                    distance_trex_detect_obstacles,
                    substraction,
                    is_need_quick_jump,
                )
                break


# начало программы
if __name__ == "__main__":
    while True:
        if keyboard.is_pressed("enter"):
            start_time = time.time()
            time.sleep(0.5)
            play_trex()
            break
