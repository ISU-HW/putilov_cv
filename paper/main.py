import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFont


def find_paper_by_corners(image):
    paper_mask = get_best_paper_candidate(image)

    if np.sum(paper_mask) == 0:
        return None

    contours, _ = cv2.findContours(
        paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    epsilons = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

    for epsilon in epsilons:
        eps_value = epsilon * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, eps_value, True)

        if len(approx) == 4:
            corners = order_points(approx.reshape(4, 2))

            if is_valid_rectangle(corners, paper_mask):
                return corners.astype(np.int32)

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    corners = order_points(box)

    return corners.astype(np.int32)


def is_valid_rectangle(corners, paper_mask):
    area = cv2.contourArea(corners.astype(np.float32))
    image_area = paper_mask.shape[0] * paper_mask.shape[1]
    size_ratio = area / image_area

    if size_ratio < 0.01 or size_ratio > 0.9:
        return False

    rect = order_points(corners)
    min_distance = 20

    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(rect[i] - rect[j])
            if dist < min_distance:
                return False

    mask = np.zeros(paper_mask.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)

    intersection = cv2.bitwise_and(paper_mask, mask)
    intersection_area = np.sum(intersection > 0)
    mask_area = np.sum(mask > 0)

    overlap_ratio = intersection_area / mask_area if mask_area > 0 else 0

    return overlap_ratio > 0.3


def check_rectangularity(corners):
    ordered_corners = order_points(corners)

    angles = []
    for i in range(4):
        p1 = ordered_corners[i]
        p2 = ordered_corners[(i + 1) % 4]
        p3 = ordered_corners[(i + 2) % 4]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_angle)))
        angles.append(abs(90 - angle))

    avg_deviation = np.mean(angles)
    rectangularity = max(0.0, 1.0 - avg_deviation / 30.0)

    return rectangularity


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def create_text_image(text, size):
    img = Image.new("RGBA", size, color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    try:
        font_size = max(24, min(size) // 8)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font_size = max(24, min(size) // 8)
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font_size = max(20, min(size) // 10)
            font = ImageFont.load_default()

    char_bbox = draw.textbbox((0, 0), "А", font=font)
    char_width = char_bbox[2] - char_bbox[0]
    char_height = char_bbox[3] - char_bbox[1]

    margin = max(20, min(size) // 20)
    max_chars_per_line = max(1, (size[0] - 2 * margin) // char_width)
    line_height = int(char_height * 1.3)
    max_lines = max(1, (size[1] - 2 * margin) // line_height)

    lines = wrap_text_improved(text, max_chars_per_line, max_lines)

    total_text_height = len(lines) * line_height
    start_y = max(margin, (size[1] - total_text_height) // 2)

    for i, line in enumerate(lines):
        if i >= max_lines:
            break

        line_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = line_bbox[2] - line_bbox[0]
        x = max(margin, (size[0] - text_width) // 2)
        y = start_y + i * line_height

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text(
                        (x + dx, y + dy), line, font=font, fill=(255, 255, 255, 255)
                    )

        draw.text((x, y), line, font=font, fill=(0, 0, 0, 255))

    return np.array(img)


def wrap_text_improved(text, max_chars_per_line, max_lines):
    if not text.strip():
        return [""]

    words = text.split()
    lines = []
    current_line = ""

    for word in words:

        test_line = current_line + " " + word if current_line else word

        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:

            if current_line:
                lines.append(current_line)
                if len(lines) >= max_lines:
                    break
                current_line = word
            else:

                if len(word) > max_chars_per_line:

                    for i in range(0, len(word), max_chars_per_line):
                        part = word[i : i + max_chars_per_line]
                        lines.append(part)
                        if len(lines) >= max_lines:
                            break
                    if len(lines) >= max_lines:
                        break
                    current_line = ""
                else:
                    current_line = word

    if current_line and len(lines) < max_lines:
        lines.append(current_line)

    if not lines:
        lines = [""]

    return lines


def project_text_on_paper(image, text):
    corners = find_paper_by_corners(image)
    if corners is None:
        print("Лист не найден")
        return image

    rect = order_points(corners)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    text_img = create_text_image(text, (maxWidth, maxHeight))
    text_bgr = cv2.cvtColor(text_img[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = text_img[:, :, 3]

    dst_points = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(dst_points, rect)
    warped_text = cv2.warpPerspective(text_bgr, M, (image.shape[1], image.shape[0]))
    warped_alpha = cv2.warpPerspective(alpha, M, (image.shape[1], image.shape[0]))

    result = image.copy()

    text_mask = warped_alpha > 50

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if text_mask[y, x]:

                if cv2.pointPolygonTest(corners.astype(np.float32), (x, y), False) >= 0:

                    alpha_val = warped_alpha[y, x] / 255.0
                    result[y, x] = (1 - alpha_val) * result[
                        y, x
                    ] + alpha_val * warped_text[y, x]

    return result


def camera_mode(text):
    cap = cv2.VideoCapture(0)

    print("Управление:")
    print("q - выход")
    print("s - сохранить кадр")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = project_text_on_paper(frame, text)

        cv2.putText(
            result,
            "Press 'q' to quit, 's' to save",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Проекция текста", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"result_{frame_count}.jpg"
            cv2.imwrite(filename, result)
            print(f"Кадр сохранен как {filename}")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def get_best_paper_candidate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    candidates = []

    thresholds = [180, 190, 200, 210, 220, 230]

    for thresh in thresholds:

        _, white_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        processed_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)

        candidate = get_largest_white_region(processed_mask)
        candidates.append((candidate, f"thresh_{thresh}"))

        eroded_mask = cv2.erode(white_mask, kernel, iterations=1)
        eroded_mask = cv2.dilate(eroded_mask, kernel, iterations=2)

        candidate = get_largest_white_region(eroded_mask)
        candidates.append((candidate, f"thresh_{thresh}_eroded"))

    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    candidate = get_largest_white_region(adaptive_mask)
    candidates.append((candidate, "adaptive"))

    best_candidate = None
    best_score = 0

    for candidate_mask, method in candidates:
        score = evaluate_paper_candidate(candidate_mask, image)
        if score > best_score:
            best_score = score
            best_candidate = candidate_mask

    return best_candidate if best_candidate is not None else np.zeros_like(gray)


def evaluate_paper_candidate(mask, image):
    if np.sum(mask) == 0:
        return 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    image_area = image.shape[0] * image.shape[1]
    size_ratio = area / image_area

    if size_ratio < 0.02 or size_ratio > 0.8:
        return 0

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        shape_score = 1.0
    elif len(approx) in [3, 5, 6]:
        shape_score = 0.7
    else:
        shape_score = 0.3

    rect = cv2.minAreaRect(largest_contour)
    width, height = rect[1]
    if width > 0 and height > 0:
        aspect_ratio = max(width, height) / min(width, height)
        if 1.0 <= aspect_ratio <= 4.0:
            aspect_score = 1.0
        else:
            aspect_score = 0.5
    else:
        aspect_score = 0

    rect_area = width * height
    fill_ratio = area / rect_area if rect_area > 0 else 0
    fill_score = min(1.0, fill_ratio * 1.2)

    size_score = min(1.0, size_ratio * 8)

    total_score = (
        shape_score * 0.3 + aspect_score * 0.2 + fill_score * 0.3 + size_score * 0.2
    )

    return total_score


def get_largest_white_region(white_mask):
    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros_like(white_mask)

    largest_contour = max(contours, key=cv2.contourArea)

    largest_mask = np.zeros_like(white_mask)
    cv2.fillPoly(largest_mask, [largest_contour], 255)

    return largest_mask


def test_mode(text, test_path):
    test_image = cv2.imread(test_path)
    if test_image is None:
        print(f"Не удалось загрузить: {test_path}")
        return

    best_paper_region = get_best_paper_candidate(test_image)

    corners = find_paper_by_corners(test_image)

    image_with_contour = test_image.copy()
    if corners is not None:
        cv2.polylines(image_with_contour, [corners], True, (0, 0, 255), 3)

    result = project_text_on_paper(test_image, text)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Оригинал")
    ax1.axis("off")

    ax2.imshow(best_paper_region, cmap="gray")
    ax2.set_title("2. Лучший кандидат на лист")
    ax2.axis("off")

    ax3.imshow(cv2.cvtColor(image_with_contour, cv2.COLOR_BGR2RGB))
    if corners is not None:
        ax3.set_title("3. Найденные углы")
    else:
        ax3.set_title("3. Углы не найдены")
    ax3.axis("off")

    ax4.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax4.set_title("4. Результат с текстом")
    ax4.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Режим:")
    print("1. Камера")
    print("2. Тест")

    choice = input("Номер: ")
    text = input("Текст: ")

    if choice == "1":
        camera_mode(text)
    elif choice == "2":
        test_mode(text, "paper.jpg")
        test_mode(text, "paper1.jpg")
        test_mode(text, "paper2.jpg")
        test_mode(text, "paper3.jpg")
    else:
        print("Неверный выбор")
