import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

def vertical_gradient(color1, color2, size):
    image = np.zeros((size, size, 3), dtype="uint8")
    for i, v in enumerate(np.linspace(0, 1, size)):
        r = lerp(color1[0], color2[0], v)
        g = lerp(color1[1], color2[1], v)
        b = lerp(color1[2], color2[2], v)
        image[i, :, :] = [r, g, b]
    return image

def diagonal_gradient(color1, color2, size):
    image = np.zeros((size, size, 3), dtype="uint8")
    for i in range(size):
        for j in range(size):
            t = (i + j) / (2 * size)
            r = lerp(color1[0], color2[0], t)
            g = lerp(color1[1], color2[1], t)
            b = lerp(color1[2], color2[2], t)
            image[i, j, :] = [r, g, b]
    return image

size = 100
color1 = [253, 187, 45]
color2 = [34, 193, 195]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(vertical_gradient(color1, color2, size))
plt.title('Вертикальный градиент')

plt.subplot(1, 2, 2)
plt.imshow(diagonal_gradient(color1, color2, size))
plt.title('Диагональный градиент')

plt.show()
