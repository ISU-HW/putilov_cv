import numpy as np
import cv2


BINARY_THRESHOLD = 120
EROSION_ITERATIONS = 40
MIN_PENCIL_AREA = 500000
MAX_PENCIL_AREA = 700000


def preprocess_image(image):
    _, binary_image = cv2.threshold(image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    eroded_image = cv2.erode(binary_image, None, iterations=EROSION_ITERATIONS)

    processed_image = cv2.bitwise_not(eroded_image)

    return processed_image


def is_pencil(area):
    return MIN_PENCIL_AREA < area < MAX_PENCIL_AREA


def count_pencils_in_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return 0

    processed_image = preprocess_image(image)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        processed_image, connectivity=4, ltype=cv2.CV_32S
    )

    pencils_count = 0

    for component_id in range(1, num_labels):

        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        width = stats[component_id, cv2.CC_STAT_WIDTH]
        height = stats[component_id, cv2.CC_STAT_HEIGHT]
        area = stats[component_id, cv2.CC_STAT_AREA]

        center_x, center_y = centroids[component_id]

        if is_pencil(area):
            pencils_count += 1

    return pencils_count


total_pencils = 0

print("Карндашей:")
for image_number in range(1, 13):
    image_path = f"images/img ({image_number}).jpg"

    pencils_in_image = count_pencils_in_image(image_path)
    total_pencils += pencils_in_image

    print(f" {pencils_in_image} {"✏️"*pencils_in_image}  (img {image_number})")

print(f"ИТОГО карандашей на всех изображениях: {total_pencils}")
