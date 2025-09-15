import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from collections import defaultdict
from pathlib import Path


def calculate_filling_factor(region_array):
    return np.sum(region_array) / region_array.size


def count_holes_in_region_image(region_image):
    inverted_image = np.logical_not(region_image)
    labeled_holes = label(inverted_image)
    hole_regions = regionprops(labeled_holes)

    holes_count = 0
    for hole_region in hole_regions:
        coordinates = np.where(labeled_holes == hole_region.label)
        is_boundary_touching = True

        for y_coord, x_coord in zip(*coordinates):
            if (
                y_coord == 0
                or x_coord == 0
                or y_coord == labeled_holes.shape[0] - 1
                or x_coord == labeled_holes.shape[1] - 1
            ):
                is_boundary_touching = False
                break

        holes_count += is_boundary_touching
    return holes_count


def has_vertical_line(region_array, min_width=1):
    column_means = region_array.mean(axis=0)
    vertical_columns = np.sum(column_means == 1)
    return min_width <= vertical_columns


def recognize_symbol(region):
    region_image = region.image
    filling_factor = calculate_filling_factor(region_image)

    if filling_factor == 1.0:
        return "-"

    holes_count = count_holes_in_region_image(region.image)

    if holes_count == 2:

        if has_vertical_line(region_image, min_width=3):
            return "B"
        else:
            return "8"

    elif holes_count == 1:

        centroid_y_normalized = region.centroid_local[0] / region_image.shape[0]
        centroid_x_normalized = region.centroid_local[1] / region_image.shape[1]

        if np.isclose(centroid_y_normalized, centroid_x_normalized, 0.05):
            if has_vertical_line(region_image) and (
                centroid_y_normalized < 0.4 or centroid_x_normalized < 0.4
            ):
                return "P"
            else:
                return "0"
        elif has_vertical_line(region_image):
            if filling_factor > 0.53:
                return "D"
            return "P"
        else:
            if centroid_y_normalized < 0.5 or centroid_x_normalized < 0.5:
                if filling_factor > 0.5:
                    return "0"
            return "A"

    else:

        if has_vertical_line(region_image):
            if filling_factor > 0.5:
                return "*"
            return "1"
        else:
            eccentricity = region.eccentricity

            framed_image = region_image.copy()
            framed_image[0, :] = 1
            framed_image[-1, :] = 1
            framed_image[:, 0] = 1
            framed_image[:, -1] = 1

            framed_holes = count_holes_in_region_image(framed_image)

            if eccentricity < 0.4:
                return "*"
            else:
                match framed_holes:
                    case 2:
                        return "/"
                    case 4:
                        return "X"

                if eccentricity > 0.5:
                    return "W"
                else:
                    return "*"


def main():

    source_image = plt.imread("symbols.png").mean(axis=2)
    source_image[source_image > 0] = 1

    labeled_image = label(source_image)
    detected_regions = regionprops(labeled_image)

    total_symbols = len(detected_regions)
    symbol_count = defaultdict(lambda: 0)

    output_path = Path(".") / "result"
    output_path.mkdir(exist_ok=True)

    print(f"Обнаружено символов: {total_symbols}")

    plt.figure(figsize=(8, 6))
    for region_index, region in enumerate(detected_regions):
        print(f"Обрабатывается регион {region_index + 1}/{total_symbols}")

        recognized_symbol = recognize_symbol(region)

        plt.clf()
        plt.title(f"Символ: {recognized_symbol} (регион {region_index})")
        plt.imshow(region.image, cmap="gray")
        plt.tight_layout()
        plt.savefig(output_path / f"region_{region_index:03d}")

        symbol_count[recognized_symbol] += 1

    for symbol, frequency in symbol_count.items():
        print(f"Символ '{symbol}': {frequency} раз")

    print(f"\nВсего уникальных символов: {len(symbol_count)}")


if __name__ == "__main__":
    main()
