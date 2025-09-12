from skimage.measure import label
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_opening,
    binary_erosion,
)
import matplotlib.pyplot as plt
import numpy as np

images = [
    np.load("wires1npy.txt"),
    np.load("wires2npy.txt"),
    np.load("wires3npy.txt"),
    np.load("wires4npy.txt"),
    np.load("wires5npy.txt"),
    np.load("wires6npy.txt"),
]
for image in images:
    labeled_image = label(image)
    arr = []
    for i in range(1, labeled_image.max() + 1):
        img = labeled_image == i
        img1 = binary_erosion(img)
        final = label(img1)

        arr.append(final.max())

    for i in range(len(arr)):
        if arr[i] == 0:
            print(f"Провод {i+1} < 3 пикселов")
            continue

        if arr[i] == 1:
            print(f"Провод {i+1} целый")
        else:
            print(f"Провод {i+1} порван на {arr[i]} частей")
    print()

    plt.subplot(121)
    plt.imshow(image)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(labeled_image)
    plt.show()
