import cv2
import matplotlib.pyplot as plt
from collections import defaultdict


def get_color_name(hue):
    hue_360 = hue * 2

    colors = [
        (10, "Красный"),
        (40, "Оранжевый"),
        (70, "Желтый"),
        (120, "Зеленый"),
        (180, "Голубой"),
        (220, "Синий"),
        (300, "Фиолетовый"),
        (360, "Розовый"),
    ]

    for threshold, color in colors:
        if hue_360 <= threshold:
            return color
    return "Красный"


def analyze_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    circles = defaultdict(int)
    rectangles = defaultdict(int)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(gray, connectivity=4)

    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]

        if y + h > hsv.shape[0] or x + w > hsv.shape[1]:
            continue

        center_y, center_x = y + h // 2, x + w // 2
        hue = int(hsv[center_y, center_x, 0])

        if hue < 0 or hue > 179:
            continue

        area_ratio = area / (w * h)

        if area_ratio > 0.95:
            rectangles[hue] += 1
        else:
            circles[hue] += 1

    return circles, rectangles, image


def print_results(circles, rectangles):
    print("\n🔵 Круги:")
    if circles:
        for hue, count in sorted(circles.items(), key=lambda x: x[1], reverse=True):
            print(f"   {get_color_name(hue)}: {count} шт.")
        print(f"   Всего: {sum(circles.values())} шт.")
    else:
        print("   Не найдены")

    print("\n🔲 Прямоугольники:")
    if rectangles:
        for hue, count in sorted(rectangles.items(), key=lambda x: x[1], reverse=True):
            print(f"   {get_color_name(hue)}: {count} шт.")
        print(f"   Всего: {sum(rectangles.values())} шт.")
    else:
        print("   Не найдены")

    total = sum(circles.values()) + sum(rectangles.values())
    print(f"\nОбщий итог: {total} фигур")


def show_image(image):
    plt.figure(figsize=(10, 6))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Анализируемое изображение")
    plt.axis("off")
    plt.show()


def main():
    try:
        image_path = "figures_and_colors/balls_and_rects.png"
        circles, rectangles, image = analyze_image(image_path)

        print_results(circles, rectangles)
        show_image(image)

    except FileNotFoundError:
        print("Файл изображения не найден")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
