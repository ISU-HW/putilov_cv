import numpy as np
import cv2

BINARY_THRESHOLD = 120
EROSION_ITERATIONS = 30
AREA_TOLERANCE = 0.3


def preprocess_image(image):
    """Предобработка изображения для выделения объектов"""
    _, binary_image = cv2.threshold(image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    eroded_image = cv2.erode(binary_image, None, iterations=EROSION_ITERATIONS)
    processed_image = cv2.bitwise_not(eroded_image)
    return processed_image


def get_reference_pencil_area(reference_image_path):
    """Получает площадь эталонного карандаша из reference_image_path"""
    image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    processed_image = preprocess_image(image)
    output = cv2.connectedComponentsWithStats(processed_image, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    max_area = 0
    best_component = None

    for j in range(1, numLabels):
        area = stats[j, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            best_component = j

    if best_component is not None:
        width = stats[best_component, cv2.CC_STAT_WIDTH]
        height = stats[best_component, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / min(width, height)
        return max_area
    else:
        return None


def is_pencil(area, reference_area):
    """Определяет, является ли объект карандашом на основе сравнения с эталонной площадью"""
    min_area = reference_area * (1 - AREA_TOLERANCE)
    max_area = reference_area * (1 + AREA_TOLERANCE)
    return min_area <= area <= max_area


def count_pencils_in_image(image_path, reference_area):
    """Подсчитывает количество карандашей на одном изображении"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0

    processed_image = preprocess_image(image)
    output = cv2.connectedComponentsWithStats(processed_image, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    pencils_count = 0
    for j in range(1, numLabels):
        area = stats[j, cv2.CC_STAT_AREA]
        if is_pencil(area, reference_area):
            pencils_count += 1

    return pencils_count


# Получаем эталонную площадь
reference_area = get_reference_pencil_area("images/pencil.jpg")
if reference_area is None:
    print("Невозможно продолжить без эталонного изображения")
    exit(1)

print(f"Площадь эталонного карандаша: {reference_area} пикселей")
print(
    f"Диапазон поиска: {reference_area * (1 - AREA_TOLERANCE):.0f} - {reference_area * (1 + AREA_TOLERANCE):.0f} пикселей"
)

# Подсчет карандашей
total_pencils = 0
print("Карандашей:")
for image_number in range(1, 13):
    image_path = f"images/img ({image_number}).jpg"
    pencils_in_image = count_pencils_in_image(image_path, reference_area)
    total_pencils += pencils_in_image
    print(f"{pencils_in_image} {"✏️"*pencils_in_image}  (img {image_number})")
print(f"ИТОГО карандашей на всех изображениях: {total_pencils}")
