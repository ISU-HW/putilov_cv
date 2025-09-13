import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from matplotlib.colors import ListedColormap


def crop_object_to_minimal_bounds(binary_object):
    object_pixel_coordinates = np.argwhere(binary_object)
    min_row_col = object_pixel_coordinates.min(axis=0)
    max_row_col = object_pixel_coordinates.max(axis=0)
    cropped_object = binary_object[
        min_row_col[0] : max_row_col[0] + 1, min_row_col[1] : max_row_col[1] + 1
    ]
    return cropped_object


source_image = np.load("ps.npy.txt")
labeled_image = label(source_image)
total_objects_count = labeled_image.max()
unique_object_types = []
object_to_type_mapping = {}

for current_object_id in range(1, total_objects_count + 1):
    current_object = crop_object_to_minimal_bounds(labeled_image == current_object_id)
    is_duplicate_found = False

    for type_index, (template_object, occurrence_count) in enumerate(
        unique_object_types
    ):
        if current_object.shape != template_object.shape:
            continue
        if np.array_equal(template_object, current_object):
            is_duplicate_found = True
            unique_object_types[type_index][1] += 1
            object_to_type_mapping[current_object_id] = type_index + 1
            break

    if not is_duplicate_found:
        unique_object_types.append([current_object, 1])
        object_to_type_mapping[current_object_id] = len(unique_object_types)

print(f"Количество объектов: {total_objects_count}")
print(f"Типов объектов: {len(unique_object_types)}")
for type_index, (template_object, count) in enumerate(unique_object_types):
    print(f"Объект {type_index + 1}: {count} раз")

type_colored_image = np.zeros_like(labeled_image)
for object_id in range(1, total_objects_count + 1):
    if object_id in object_to_type_mapping:
        type_colored_image[labeled_image == object_id] = object_to_type_mapping[
            object_id
        ]

background_color_rgb = [1, 1, 1]
visualization_window_width = 10
visualization_window_height = 8.5

high_contrast_colors = np.array(
    [
        background_color_rgb,
        [1, 0, 0],
        [0, 0.8, 0],
        [1, 0.7, 0],
        [0, 0, 1],
        [0.8, 0, 0.8],
    ]
)

discrete_colormap = ListedColormap(high_contrast_colors)

main_figure = plt.figure(
    figsize=(visualization_window_width, visualization_window_height)
)

main_objects_plot = plt.subplot2grid((6, 3), (0, 0), colspan=3, rowspan=4)
main_objects_plot.imshow(
    type_colored_image, cmap=discrete_colormap, interpolation="nearest"
)
main_objects_plot.set_title("Полная карта объектов", fontsize=14)

unique_types_count = len(unique_object_types)
max_columns_per_row = min(5, unique_types_count)

for type_index, (object_template, occurrence_count) in enumerate(unique_object_types):
    column_position = type_index % max_columns_per_row
    row_position = 4

    small_subplot = plt.subplot2grid(
        (6, max_columns_per_row * 3),
        (row_position, column_position * 3),
        colspan=2,
        rowspan=1,
    )

    object_visualization = np.zeros((*object_template.shape, 3))
    object_visualization[object_template == 0] = background_color_rgb

    type_color_index = type_index + 1
    if type_color_index < len(high_contrast_colors):
        object_visualization[object_template == 1] = high_contrast_colors[
            type_color_index
        ]
    else:
        object_visualization[object_template == 1] = [0, 0, 0]

    small_subplot.imshow(object_visualization)
    small_subplot.set_title("")
    small_subplot.set_xlabel("")
    small_subplot.axis("off")

    small_subplot.text(
        0.5,
        -0.2,
        f"{occurrence_count} раз",
        horizontalalignment="center",
        verticalalignment="top",
        transform=small_subplot.transAxes,
        fontsize=10,
        fontweight="bold",
        color="black",
    )

plt.figtext(
    0.5,
    0.05,
    f"Всего: {total_objects_count}",
    horizontalalignment="center",
    fontsize=12,
    fontweight="bold",
)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.15)
plt.show()
