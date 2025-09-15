import numpy as np
import cv2
from collections import Counter


def count_holes_in_image(binary_image):
    prepared_image = (binary_image > 0).astype(np.uint8) * 255

    # Все контуры и их вложенность
    contours, hierarchy_info = cv2.findContours(
        prepared_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy_info is None or len(contours) == 0:
        return []

    figures_with_holes = []

    for contour_index in range(len(contours)):

        contour_hierarchy = get_contour_info(hierarchy_info, contour_index)

        if is_outer_contour(contour_hierarchy):

            holes_count = count_holes_inside_contour(hierarchy_info, contour_index)
            figures_with_holes.append(holes_count)

    return figures_with_holes


def get_contour_info(hierarchy_info, contour_index):
    raw_info = hierarchy_info[0][contour_index]

    return {
        "next_sibling": raw_info[0],
        "prev_sibling": raw_info[1],
        "first_child": raw_info[2],
        "parent": raw_info[3],
    }


def is_outer_contour(contour_info):
    return contour_info["parent"] == -1


def count_holes_inside_contour(hierarchy_info, outer_contour_index):
    outer_contour_info = get_contour_info(hierarchy_info, outer_contour_index)

    current_hole = outer_contour_info["first_child"]

    holes_count = 0

    while current_hole != -1:
        holes_count += 1

        next_hole_info = get_contour_info(hierarchy_info, current_hole)
        current_hole = next_hole_info["next_sibling"]

    return holes_count


def analyze_holes_file(filename="holes.npy"):
    try:

        data = np.load(filename)
        print(f"Загружен файл: {filename}")
        print(f"Размерность: {data.shape}, Тип: {data.dtype}")

        all_hole_counts = []

        if len(data.shape) == 3:

            for i in range(data.shape[0]):
                holes = count_holes_in_image(data[i])
                all_hole_counts.extend(holes)
                print(
                    f"Изображение {i+1}: найдено {len(holes)} фигур с отверстиями: {holes}"
                )

        elif len(data.shape) == 2:

            holes = count_holes_in_image(data)
            all_hole_counts.extend(holes)
            print(f"Найдено {len(holes)} фигур с отверстиями: {holes}")

        counter = Counter(all_hole_counts)

        print(f"Всего фигур: {len(all_hole_counts)}")

        for holes_num in sorted(counter.keys()):
            count = counter[holes_num]
            percentage = (count / len(all_hole_counts)) * 100 if all_hole_counts else 0
            print(f"Фигур с {holes_num} отверстиями: {count} ({percentage:.1f}%)")

        return counter

    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    result = analyze_holes_file("holes.npy")
