import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

size = 3
image = np.zeros((size, size, 3), dtype="uint8")
assert image.shape[0] == image.shape[1]

color1 = [253, 187, 45]
color2 = [34, 193, 195]

for i, v in enumerate(np.linspace(0, 1, image.shape[0])):
    r = lerp(color1[0], color2[0], v)
    g = lerp(color1[1], color2[1], v)
    b = lerp(color1[2], color2[2], v)
    image[i, :, :] = [r, g, b]

image = np.rot90(image, k=3)

plt.figure(1)
plt.imshow(image)
plt.show()