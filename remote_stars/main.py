import numpy as np
from PIL import Image


def is_valid_point(image_array, row, col):
    if row < 0 or row >= image_array.shape[0]:
        return False
    if col < 0 or col >= image_array.shape[1]:
        return False
    if image_array[row, col] == 0:
        return False
    return True


def get_neighbors(image_array, row, col):
    neighbors = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if is_valid_point(image_array, nr, nc):
            neighbors.append((nr, nc))
    return neighbors


def find_extremum_points(image_array):
    extremums = []
    for row in range(image_array.shape[0]):
        for col in range(image_array.shape[1]):
            if not is_valid_point(image_array, row, col):
                continue
            neighbors = get_neighbors(image_array, row, col)
            if not neighbors:
                continue
            current_value = image_array[row, col]
            is_max = True
            for nr, nc in neighbors:
                if current_value <= image_array[nr, nc]:
                    is_max = False
                    break
            if is_max:
                extremums.append((row, col))
    return extremums


def calculate_distance(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def load_image(filename):
    img = Image.open(filename)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)


def analyze_image(filename):
    image_array = load_image(filename)
    extremums = find_extremum_points(image_array)
    if len(extremums) == 2:
        return calculate_distance(extremums[0], extremums[1])
    return None


if __name__ == "__main__":
    result = analyze_image("test1.png")
    if result:
        print(f"Distance: {result}")
    else:
        print("No two extremums found")
