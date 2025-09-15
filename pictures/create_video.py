import cv2
import os
import random
from PIL import Image
import numpy as np


def create_video_from_images(images_folder="images", output_file="input.avi", fps=30):
    if not os.path.exists(images_folder):
        print(f"Папка {images_folder} не найдена!")
        return

    image_extensions = (".jpg", ".jpeg", ".png")

    image_files = []
    for file in os.listdir(images_folder):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(images_folder, file))

    if not image_files:
        print("В папке не найдены изображения!")
        return

    print(f"Найдено изображений: {len(image_files)}")

    frame_sequence = []

    for image_path in image_files:

        repeat_count = random.randint(100, 1000)
        print(f"Изображение {os.path.basename(image_path)}: {repeat_count} повторений")

        for _ in range(repeat_count):
            frame_sequence.append(image_path)

    random.shuffle(frame_sequence)

    print(f"Общее количество кадров: {len(frame_sequence)}")
    print(f"Продолжительность видео: {len(frame_sequence) / fps:.2f} секунд")

    first_image = Image.open(frame_sequence[0])
    width, height = first_image.size

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for i, image_path in enumerate(frame_sequence):
        if i % 100 == 0:
            print(f"Обработано кадров: {i}/{len(frame_sequence)}")

        image = Image.open(image_path)

        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)

        if image.mode == "RGBA":
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Видео успешно создано: {output_file}")


if __name__ == "__main__":
    create_video_from_images()
