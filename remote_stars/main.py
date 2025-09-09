import unittest
import numpy as np
from PIL import Image


def is_valid_point(image_array, row, col):
    if not 0 <= col < image_array.shape[1]:
        return False
    if not 0 <= row < image_array.shape[0]:
        return False
    if image_array[row, col] != 0:
        return True
    return False


def get_neighbors(image_array, row, col):
    left = row, col - 1
    right = row, col + 1
    top = row - 1, col
    bottom = row + 1, col
    valid_neighbors = []
    for neighbor in [left, right, top, bottom]:
        if is_valid_point(image_array, *neighbor):
            valid_neighbors.append(neighbor)
    return valid_neighbors


def find_extremum_points(image_array):
    extremum_points = []
    for row in range(image_array.shape[0]):
        for col in range(image_array.shape[1]):
            if not is_valid_point(image_array, row, col):
                continue
            neighbors = get_neighbors(image_array, row, col)
            if len(neighbors) == 0:
                continue
            is_extremum = True
            for neighbor_row, neighbor_col in neighbors:
                if image_array[row, col] <= image_array[neighbor_row, neighbor_col]:
                    is_extremum = False
                    break
            if is_extremum:
                extremum_points.append((row, col))
    return extremum_points


def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def load_image(filename):
    image = Image.open(filename)
    if image.mode != "L":
        image = image.convert("L")
    return np.array(image, dtype=np.uint8)


def analyze_image(filename):
    image_array = load_image(filename)
    extremum_points = find_extremum_points(image_array)
    if len(extremum_points) == 2:
        distance = calculate_distance(extremum_points[0], extremum_points[1])
        return distance
    else:
        return None


class TestImageAnalysis(unittest.TestCase):

    def setUp(self):
        self.test_array = np.array([[0, 1, 0], [1, 5, 1], [0, 1, 0]], dtype=np.uint8)

        self.two_peaks_array = np.array(
            [[0, 1, 0, 1, 0], [1, 5, 1, 3, 1], [0, 1, 0, 1, 0]], dtype=np.uint8
        )

        self.empty_array = np.zeros((3, 3), dtype=np.uint8)

    def test_is_valid_point(self):
        self.assertTrue(is_valid_point(self.test_array, 1, 1))
        self.assertTrue(is_valid_point(self.test_array, 0, 1))
        self.assertFalse(is_valid_point(self.test_array, -1, 1))
        self.assertFalse(is_valid_point(self.test_array, 3, 1))
        self.assertFalse(is_valid_point(self.test_array, 0, 0))

    def test_get_neighbors(self):
        neighbors = get_neighbors(self.test_array, 1, 1)
        expected_neighbors = [(1, 0), (1, 2), (0, 1), (2, 1)]
        self.assertEqual(set(neighbors), set(expected_neighbors))

        neighbors = get_neighbors(self.empty_array, 1, 1)
        self.assertEqual(neighbors, [])

    def test_find_extremum_points(self):
        extremums = find_extremum_points(self.test_array)
        self.assertEqual(len(extremums), 1)
        self.assertIn((1, 1), extremums)

        extremums = find_extremum_points(self.two_peaks_array)
        self.assertEqual(len(extremums), 2)

        extremums = find_extremum_points(self.empty_array)
        self.assertEqual(len(extremums), 0)

    def test_calculate_distance(self):
        distance = calculate_distance((0, 0), (3, 4))
        self.assertAlmostEqual(distance, 5.0, places=2)

        distance = calculate_distance((1, 1), (1, 1))
        self.assertEqual(distance, 0.0)

    def test_analyze_image_files(self):
        test_images = [
            "test1.png",
            "test2.png",
            "test3.png",
            "test4.png",
            "test_empty.png",
            "test_noisy.png",
            "test_one_peak.png",
            "test_three_peaks.png",
        ]

        results = {}
        for img in test_images:
            try:
                result = analyze_image(img)
                results[img] = result
                self.assertIsInstance(result, (float, type(None)))
            except FileNotFoundError:
                results[img] = "File not found"
            except Exception as e:
                results[img] = f"Error: {e}"

        print("\nResults:")
        for img, result in results.items():
            if isinstance(result, float):
                print(f"{img}: {result:.1f}")
            else:
                print(f"{img}: {result}")

    def test_empty_image(self):
        try:
            result = analyze_image("test_empty.png")
            self.assertIsNone(result)
        except FileNotFoundError:
            self.skipTest("test_empty.png not found")

    def test_noisy_image(self):
        try:
            result = analyze_image("test_noisy.png")
            self.assertIsNone(result)
        except FileNotFoundError:
            self.skipTest("test_noisy.png not found")

    def test_one_peak_image(self):
        try:
            result = analyze_image("test_one_peak.png")
            self.assertIsNone(result)
        except FileNotFoundError:
            self.skipTest("test_one_peak.png not found")

    def test_three_peaks_image(self):
        try:
            result = analyze_image("test_three_peaks.png")
            self.assertIsNone(result)
        except FileNotFoundError:
            self.skipTest("test_three_peaks.png not found")

    def test_two_peak_images(self):
        two_peak_images = ["test1.png", "test2.png", "test3.png", "test4.png"]
        for img in two_peak_images:
            try:
                result = analyze_image(img)
                self.assertIsNotNone(result)
                self.assertIsInstance(result, float)
                self.assertGreater(result, 0)
            except FileNotFoundError:
                self.skipTest(f"{img} not found")

    def test_edge_cases(self):
        single_point = np.array([[5]], dtype=np.uint8)
        extremums = find_extremum_points(single_point)
        self.assertEqual(len(extremums), 0)

        small_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        extremums = find_extremum_points(small_array)
        self.assertEqual(len(extremums), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
