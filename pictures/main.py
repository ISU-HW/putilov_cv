import cv2
import numpy as np
from PIL import Image
import os


def load_reference_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден!")

    image = Image.open(image_path).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def compare_images_mse(img1, img2, threshold=1000):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return mse <= threshold


def analyze_video(video_path, reference_image_path, threshold=1000):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео файл {video_path} не найден!")

    reference_image = load_reference_image(reference_image_path)
    print(f"Эталон загружен: {reference_image.shape}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Кадров: {total_frames}, FPS: {fps:.1f}, Длительность: {total_frames/fps:.1f}с"
    )

    match_count = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape != reference_image.shape:
            frame = cv2.resize(
                frame, (reference_image.shape[1], reference_image.shape[0])
            )

        if compare_images_mse(frame, reference_image, threshold):
            match_count += 1

        frame_number += 1

        if frame_number % 1000 == 0:
            print(
                f"Обработано: {frame_number}/{total_frames}, совпадений: {match_count}"
            )

    cap.release()

    percentage = (match_count / frame_number) * 100 if frame_number > 0 else 0
    print(
        f"\nРезультат: {match_count} совпадений из {frame_number} кадров ({percentage:.2f}%)"
    )

    return match_count


if __name__ == "__main__":
    video_file = "input.avi"
    reference_image = "images/own.png"

    try:
        matches = analyze_video(video_file, reference_image)
        print(f"Найдено {matches} совпадений")
    except Exception as e:
        print(f"Ошибка: {e}")
