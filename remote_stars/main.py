import numpy as np
from PIL import Image


def is_valid_point(image_array, row, col):
    if not 0 <= col < image_array.shape[1]:
        return False
    if not 0 <= row < image_array.shape[0]:
        return False
    if image_array[row, col] != 0:
        return True
    return False


def get_neighbors(image_array, row, col):
    left = row, col - 1
    right = row, col + 1
    top = row - 1, col
    bottom = row + 1, col
    valid_neighbors = []
    for neighbor in [left, right, top, bottom]:
        if is_valid_point(image_array, *neighbor):
            valid_neighbors.append(neighbor)
    return valid_neighbors


def find_extremum_points(image_array):
    extremum_points = []
    for row in range(image_array.shape[0]):
        for col in range(image_array.shape[1]):
            if not is_valid_point(image_array, row, col):
                continue
            neighbors = get_neighbors(image_array, row, col)
            if len(neighbors) == 0:
                continue
            is_extremum = True
            for neighbor_row, neighbor_col in neighbors:
                if image_array[row, col] <= image_array[neighbor_row, neighbor_col]:
                    is_extremum = False
                    break
            if is_extremum:
                extremum_points.append((row, col))
    return extremum_points


def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def load_image(filename):
    image = Image.open(filename)
    if image.mode != "L":
        image = image.convert("L")
    return np.array(image, dtype=np.uint8)


def analyze_image(filename):
    image_array = load_image(filename)
    extremum_points = find_extremum_points(image_array)
    if len(extremum_points) == 2:
        distance = calculate_distance(extremum_points[0], extremum_points[1])
        return distance
    else:
        return None


if __name__ == "__main__":
    try:
        result = analyze_image("test1.png")
        if result is not None:
            print(f"Distance: {result:.1f}")
        else:
            print("No two extremums found")
    except FileNotFoundError:
        print("Файл не найден")
    except Exception as e:
        print(f"Ошибка: {e}")
