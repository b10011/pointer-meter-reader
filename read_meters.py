# --------------------------------- LICENCE --------------------------------- #
# Copyright (c) 2022 Niko Järvinen                                            #
#                                                                             #
# The above copyright notice and this license text shall be included as-is in #
# all copies of this software, modified or not.                               #
#                                                                             #
# This software or any part of it may be used for non-commercial purposes     #
# only.                                                                       #
#                                                                             #
# Wherever Mozilla Public License 2.0 does not contradict this license text,  #
# Mozilla Public License 2.0 shall be followed.                               #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
# --------------------------------------------------------------------------- #


# --------------------------------- IMPORTS --------------------------------- #
from PIL import Image
from enum import Enum
from pathlib import Path
from skimage.filters import threshold_multiotsu
import cv2
import json
import numpy as np
import time
from skimage.morphology import skeletonize
from skimage.measure import ransac, LineModelND
from loguru import logger

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# ---------------------------------- PATHS ---------------------------------- #
scriptdir = Path(__file__).parent.expanduser().resolve().absolute()
datasetdir = scriptdir / "dataset"
meterannotationdir = scriptdir / "annotated_templates"
# ----------------------------- GLOBAL VARIABLES ---------------------------- #

sift = cv2.SIFT_create()

# -------------------------- COMBINATION UTILITIES -------------------------- #
def multiply(values):
    res = 1
    for i in values:
        res *= i
    return res


def combinationcount(valuelists):
    lengths = [len(i) for i in valuelists]
    count = multiply(lengths)
    return count


def itemcombinations(valuelists):
    lengths = [len(i) for i in valuelists]
    count = multiply(lengths)

    divisors = [multiply(lengths[i:]) for i in range(1, len(lengths))] + [1]

    for pointer in range(count):
        res = []
        for values, divisor in zip(valuelists, divisors):
            index, pointer = divmod(pointer, divisor)
            res.append(values[index])
        yield res


def iterranges(ranges):
    yield from itemcombinations(
        [list(i) if isinstance(i, range) else list(range(i)) for i in ranges]
    )


# ---------------------------- COLOR SEGMENTATION --------------------------- #
class COLOR(Enum):
    UNCHANGED = "UNCHANGED"
    BGR2GRAY3C = "BGR2GRAY3C"
    GRAY3C2BGR = "GRAY3C2BGR"


def cvtColor(image, conversion, dst=None):
    if conversion == COLOR.UNCHANGED and dst is None:
        return image.copy()
    if conversion == COLOR.UNCHANGED and dst is not None:
        return

    elif conversion == COLOR.BGR2GRAY3C and dst is None:
        return cv2.cvtColor(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
    elif conversion == COLOR.BGR2GRAY3C and dst is not None:
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=dst)
        cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR, dst=dst)

    elif conversion == COLOR.GRAY3C2BGR and dst is None:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif conversion == COLOR.GRAY3C2BGR and dst is not None:
        cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR, dst=dst)

    elif dst is None:
        return cv2.cvtColor(image, conversion)
    else:
        cv2.cvtColor(image, conversion, dst=dst)


def colorseg_segment(image, threshold_counts):
    assert (
        len(threshold_counts) == image.shape[2]
    ), "Invalid number of thresholds"

    res = np.zeros(image.shape, dtype=np.uint8)

    all_thresholds = []

    for i in range(image.shape[2]):
        thresholds = threshold_multiotsu(
            image[:, :, i], classes=threshold_counts[i]
        )
        res[:, :, i] = np.digitize(image[:, :, i], bins=thresholds)
        all_thresholds.append(thresholds)

    return res, all_thresholds


def colorseg_visualize(digitized, threshold_counts=None):
    res = np.zeros(digitized.shape, dtype=np.uint8)

    for i in range(digitized.shape[2]):
        if threshold_counts is not None:
            max_value = threshold_counts[i] - 1
        else:
            max_value = digitized[:, :, i].max()

        if max_value > 0:
            res[:, :, i] = (digitized[:, :, i] * (255 / max_value)).round()
        else:
            res[:, :, i] = 0

    return res


def colorseg_create_mask(digitized, positives):
    res = np.zeros(digitized.shape[:-1], dtype=np.uint8)

    for positive in positives:
        coords = np.where(np.all(digitized == positive, axis=2))
        res[coords] = 255

    return res


def threshold(result, image=None, visual_segmentation=False):
    if image is None:
        image = result.transformed()

    meta = result.template.meta

    _, from_bgr, to_bgr = meta["colorspace"]
    threshold_counts = [
        meta["segments_channel_1"],
        meta["segments_channel_2"],
        meta["segments_channel_3"],
    ]
    colorstates = meta["colorstates"]

    converted = cvtColor(image, from_bgr)

    try:
        digitized, _ = colorseg_segment(converted, threshold_counts)

        if visual_segmentation:
            visualized = cvtColor(colorseg_visualize(digitized), to_bgr)
        else:
            visualized = None

        colors_digitized = [
            item
            for i, item in enumerate(iterranges(threshold_counts))
            if i in colorstates
        ]

        mask = colorseg_create_mask(digitized, colors_digitized)

        return mask, visualized

    except ValueError:
        width, height = converted.shape[1::-1]
        visualized = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)

        return mask, visualized


# ------------------------------ SIFT UTILITIES ----------------------------- #
def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def find_homography(keypointdescriptors1, keypointdescriptors2, mask=None):
    keypoints1, descriptors1 = keypointdescriptors1
    keypoints2, descriptors2 = keypointdescriptors2

    keypoint_pairs = cv2.BFMatcher(cv2.NORM_L2, False).knnMatch(
        descriptors1, trainDescriptors=descriptors2, k=2
    )

    matched_keypoint1s, matched_keypoint2s = np.float32(
        [
            (keypoints1[a.queryIdx].pt, keypoints2[a.trainIdx].pt)
            for a, b in keypoint_pairs
            if a.distance < b.distance * 0.75
        ]
    ).transpose(1, 0, 2)

    homography, mask = cv2.findHomography(
        matched_keypoint1s, matched_keypoint2s, cv2.RANSAC, 5.0, mask=mask
    )

    return homography, mask


def perspective_transform(image, corners, width, height):
    corners = corners.astype(np.float32)
    coordinates = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
    )
    perspective_matrix = cv2.getPerspectiveTransform(corners, coordinates)
    return cv2.warpPerspective(image, perspective_matrix, (width, height))


class DetectionResult:
    def __init__(self, image, template, homography):
        self.image = image
        self.template = template
        self.homography = homography

    def transformed(self, image=None, maximum_size=True, size_scaler=1):
        if image is None:
            image = self.image

        template_width, template_height = self.template.image.shape[1::-1]

        widths, heights = self.get_detection_size()

        if maximum_size:
            width = max(widths)
            height = max(heights)

            width_ratio = width / template_width
            height_ratio = height / template_height

            ratio = max(width_ratio, height_ratio)

            width = round(ratio * template_width * size_scaler)
            height = round(ratio * template_height * size_scaler)
        else:
            width = min(widths)
            height = min(heights)

            width_ratio = width / template_width
            height_ratio = height / template_height

            ratio = min(width_ratio, height_ratio)

            width = round(ratio * template_width * size_scaler)
            height = round(ratio * template_height * size_scaler)

        corners_transformed = self.convert_coordinates(
            np.float32(
                [
                    [
                        [0, 0],
                        [template_width, 0],
                        [template_width, template_height],
                        [0, template_height],
                    ]
                ]
            )
        )
        image_warped = perspective_transform(
            image, corners_transformed, width, height
        )
        return image_warped

    def get_corners(self):
        template_width, template_height = self.template.image.shape[1::-1]

        return self.convert_coordinates(
            np.float32(
                [
                    [
                        [0, 0],
                        [template_width, 0],
                        [template_width, template_height],
                        [0, template_height],
                    ]
                ]
            )
        )

    def get_detection_size(self):
        template_width, template_height = self.template.image.shape[1::-1]

        (
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ) = self.get_corners()

        top = euclidean_distance(top_left, top_right)
        right = euclidean_distance(top_right, bottom_right)
        bottom = euclidean_distance(bottom_right, bottom_left)
        left = euclidean_distance(bottom_left, top_left)

        return (top, bottom), (left, right)

    def convert_coordinates(self, coordinates):
        coordinates = np.int32(
            cv2.perspectiveTransform(coordinates, self.homography),
        ).reshape(-1, 2)

        return coordinates

    def draw_detection(
        self, image=None, color=(255, 0, 0), thickness=3, **kwargs
    ):
        if image is None:
            image = self.image.copy()

        template_width, template_height = self.template.image.shape[1::-1]

        corners_transformed = self.convert_coordinates(
            np.float32(
                [
                    [
                        [0, 0],
                        [template_width, 0],
                        [template_width, template_height],
                        [0, template_height],
                    ]
                ]
            )
        )

        cv2.polylines(
            image,
            [corners_transformed],
            True,
            color=color,
            thickness=thickness,
            **kwargs
        )

        return image


# ------------------------- METER READING UTILITIES ------------------------- #
def load_template_image(path):
    path = Path(path)
    jsonpath = path.with_suffix(".json")
    if jsonpath.exists() and jsonpath.is_file():
        with jsonpath.open("r") as f:
            meta = dict(json.load(f))
    else:
        pil_image = Image.open(str(path))
        if "_annotation" not in pil_image.info:
            raise Exception("Template image is not annotated")
        meta = dict(json.loads(pil_image.info["_annotation"]))
        pil_image.close()

    image = cv2.imread(str(path))
    image = image[
        meta["crop_top"] : meta["crop_top"] + meta["crop_height"],
        meta["crop_left"] : meta["crop_left"] + meta["crop_width"],
    ]

    keypoint_descriptors = sift.detectAndCompute(image, None)

    return image, keypoint_descriptors, meta


class TemplateImage:
    def __init__(self, imagepath):
        self.imagepath = imagepath

        (
            self.image,
            self.template_keypoints_descriptors,
            self.meta,
        ) = load_template_image(self.imagepath)

    def detect(self, image=None, keypoints_descriptors=None):
        assert (
            image is not None or keypoints_descriptors is not None
        ), "Image or keypoint-descriptors must be provided"

        if keypoints_descriptors is None:
            keypoints_descriptors = sift.detectAndCompute(image, None)

        homography, mask = find_homography(
            self.template_keypoints_descriptors, keypoints_descriptors
        )

        return DetectionResult(
            image=image, template=self, homography=homography
        )


class MeterReader:
    def __init__(self, processor):
        self.processor = processor

        self.keypoints_descriptors = None
        self.templates = dict()
        self.corners = dict()

    def add_template(self, name, template):
        if isinstance(template, (str, Path)):
            template = TemplateImage(template)

        self.templates[name] = template

    def remove_template(self, name):
        del self.templates[name]

    def find_templates(self, image):
        self.keypoints_descriptors = sift.detectAndCompute(image, None)

        self.corners.clear()

        for name, template in self.templates.items():
            detectionresult = template.detect(
                keypoints_descriptors=self.keypoints_descriptors
            )
            corners = detectionresult.get_corners()
            self.corners[name] = corners

    def detect(self, image, visual_segmentation=False, draw_detection=False):
        readings = dict()

        for template_name, template in self.templates.items():
            duration_find_templates = -time.time()
            result = template.detect(
                keypoints_descriptors=self.keypoints_descriptors
            )
            duration_find_templates += time.time()

            duration_transform = -time.time()
            cropped = result.transformed(
                image, maximum_size=False, size_scaler=1.0
            )
            duration_transform += time.time()

            duration_thresholding = -time.time()
            mask, segmentation = threshold(
                result, cropped, visual_segmentation=visual_segmentation
            )
            duration_thresholding += time.time()

            if draw_detection:
                visualized_image = image.copy()
                result.draw_detection(
                    visualized_image, color=(255, 0, 0), thickness=5
                )
            else:
                visualized_image = None

            value, angle, direction, points, lines, durations = self.processor(
                result, mask, return_verbose=True
            )

            durations = [
                ("find_templates", duration_find_templates),
                ("transform", duration_transform),
                ("thresholding", duration_thresholding),
                *durations,
            ]

            readings[template_name] = (
                value,
                result,
                points,
                lines,
                direction,
                [visualized_image, cropped, segmentation, mask],
                durations,
            )

        return readings


# ------------------ METER READER IMPLEMENTATION UTILITIES ------------------ #
def line_point_distance(linepoint, direction, point):
    lp1_x, lp1_y = linepoint
    uv_x, uv_y = direction
    p_x, p_y = point
    return abs((lp1_y - p_y) * uv_x - (lp1_x - p_x) * uv_y)


def filter_contours(contours, minimum_size=0, maximum_size=float("inf")):
    return [
        contour
        for contour in contours
        if minimum_size <= cv2.contourArea(contour) <= maximum_size
    ]


def mask_remove_small_contours(mask, minimum_size, copy=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    small_contours = filter_contours(contours, maximum_size=minimum_size)
    if copy:
        cleaned = mask.copy()
        cv2.drawContours(cleaned, small_contours, -1, 0, -1)
        return cleaned
    else:
        cv2.drawContours(mask, small_contours, -1, 0, -1)
        return mask


def mask_get_edge_points(mask, minimum_size=0, maximum_size=float("inf")):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = filter_contours(
        contours, minimum_size=minimum_size, maximum_size=maximum_size
    )
    if len(contours) == 0:
        return np.array([])
    else:
        return np.vstack([contour.reshape(-1, 2) for contour in contours])


def find_lines(points):
    points = points.reshape(-1, 2)

    while len(points) >= 3:
        model, inliers = ransac(
            points,
            LineModelND,
            min_samples=2,
            residual_threshold=2,
            max_trials=100,
        )

        yield (model.params, points[inliers])

        points = points[~inliers]


def find_n_lines_near_point(
    points, line_count, point, max_distance, min_inliers=0
):
    lines_found = 0

    for (linepoint, unitvector), inliers in find_lines(points):
        if len(inliers) >= min_inliers:
            distance = line_point_distance(linepoint, unitvector, point)
            if distance <= max_distance:
                yield ((linepoint, unitvector), inliers)
                lines_found += 1
                if lines_found == line_count:
                    break


def longest_vector_combinations(vector1, vector2, normalize=False):
    combination1 = vector1 + vector2
    combination2 = vector1 - vector2

    distance1 = np.sqrt((combination1 ** 2).sum())
    distance2 = np.sqrt((combination2 ** 2).sum())

    if normalize:
        combination1 = combination1 / distance1
        combination2 = combination2 / distance2

    if distance1 > distance2:
        return combination1, -combination1
    else:
        return combination2, -combination2


def thin(mask):
    return (
        skeletonize((mask / 255).astype(np.uint8), method="zhang") * 255
    ).astype(np.uint8)


def calculate_sector_size(angle_from, angle_to, clockwise=False):
    difference = (angle_to - angle_from) % 360

    if clockwise:
        difference = 360 - difference

    return difference


def get_full_rotation_value(
    angle_start,
    angle_stop,
    value_start,
    value_stop,
):
    # Calculate scale and non-scale sector sizes
    scale_size = calculate_sector_size(angle_start, angle_stop, True)

    # If full 360 degree scale would be used, how large difference would be
    # with the smallest and largest scale value
    scale_value_range = 360 / scale_size * (value_stop - value_start)

    return scale_value_range


def map_angle_to_value(
    angle,
    angle_start,
    angle_stop,
    value_start,
    value_stop,
    force_limits=True,
    directional=True,
):
    # Calculate scale and non-scale sector sizes
    scale_size = calculate_sector_size(angle_start, angle_stop, True)
    nonscale_size = calculate_sector_size(angle_start, angle_stop, False)

    # Calculate the middle point of the non-scale sector
    nonscale_middle = (angle_start + nonscale_size / 2) % 360

    # Calculate how much the angles must be shifted to force the non-scale
    # middle angle to be at angle 0
    shift = calculate_sector_size(nonscale_middle, 0, False)

    # Add the shift angle to the previous angles
    angle = (angle + shift) % 360
    angle_start = (angle_start + shift) % 360
    angle_stop = (angle_stop + shift) % 360
    # nonscale_middle = (nonscale_middle + shift) % 360  # TODO: REMOVE

    # Store the pointer angle in `angle1`
    angle1 = angle

    # Calculate the relative ratio on the scale
    ratio1 = (angle1 - nonscale_size / 2) / scale_size

    # Calculate the value on the scale
    value1 = (value_start - value_stop) * ratio1 + value_stop

    # If scale min and max limits should be enforced and the value is outside
    # of the range, set it to the closest allowed value
    if force_limits:
        value_min = min(value_start, value_stop)
        value_max = max(value_start, value_stop)

        value1 = max(value1, value_min)
        value1 = min(value1, value_max)

    if not directional:
        # Ensure tha angle is within [0,360)
        angle2 = (angle + 180) % 360

        # Calculate the relative ratio of the scale
        ratio2 = (angle2 - nonscale_size / 2) / scale_size

        # Calculate the value on the scale
        value2 = (value_start - value_stop) * ratio2 + value_stop

        # If scale min and max limits should be enforced and the value is
        # outside of the range, set it to the closest allowed value
        if force_limits:
            value2 = max(value2, value_min)
            value2 = min(value2, value_max)

        return value1, value2
    else:
        return value1


# ---------------------- METER READER IMPLEMENTATION 1 ---------------------- #
def implementation_1(detection, mask, return_verbose=False):
    timestamp_start_init = time.time()

    # Extract values
    width, height = mask.shape[1::-1]
    meta = detection.template.meta

    template_width = meta["crop_width"]
    template_height = meta["crop_height"]
    scale_center_x = meta["center_x"] * (width / template_width)
    scale_center_y = meta["center_y"] * (height / template_height)

    scale_center = np.array([scale_center_x, scale_center_y])

    scale_angle_1 = meta["scale_angle_1"]
    scale_value_1 = meta["scale_value_1"]
    scale_angle_2 = meta["scale_angle_2"]
    scale_value_2 = meta["scale_value_2"]

    # Calculate minimum required area for the pointer
    minimum_area = width * height * 0.004

    timestamp_start_contour_cleanup = time.time()

    # Remove too small contours
    mask_remove_small_contours(mask, minimum_area)

    timestamp_start_contour_thinning = time.time()

    # Thin the mask
    thinned = thin(mask)

    timestamp_start_line_finding = time.time()

    # Find points on the skeletons
    ys, xs = np.where(thinned)
    points = np.array(list(zip(xs, ys)))

    # Find a line from the set of points.
    # Distance from a line must be at most 10% of `min(width, height)` of the
    # mask.
    try:
        found_line = next(
            find_n_lines_near_point(
                points,
                1,
                (scale_center_x, scale_center_y),
                min(width, height) * 0.1,
            )
        )
    except StopIteration:
        if return_verbose:
            timestamp_stop = time.time()
            timestamps = [
                ("init", timestamp_start_init),
                ("contour_cleanup", timestamp_start_contour_cleanup),
                ("contour_thinning", timestamp_start_contour_thinning),
                ("line_finding", timestamp_start_line_finding),
                ("end", timestamp_stop),
            ]
            durations = [
                (name, next_timestamp - prev_timestamp)
                for (name, prev_timestamp), (_, next_timestamp) in zip(
                    timestamps, timestamps[1:]
                )
            ]
            return (None, None, None, points, [], durations)
        else:
            return None

    # Get the unit vector that is parallel to the pointer line and the RANSAC
    # inliers
    (_, pointer_direction), inliers = found_line

    timestamp_start_pointer_calculation = time.time()

    # Because the pointer direction unit vector is non-directional, both
    # directions are considered and the one with smaller distance to the
    # inliers' mean coordinate is considered to be the tip of the pointer.

    # Calculate the mean coordinate for the points that were used to define the
    # line by RANSAC
    inlier_mean = inliers.mean(axis=0)

    # Calculate the distance from inliers' mean coordinate to the unit vector
    # that originates from the scale center.
    distance_to_combined1 = euclidean_distance(
        inlier_mean, scale_center + pointer_direction
    )
    # Calculate the distance from inliers' mean coordinate to the inverse unit
    # vector that originates from the scale center.
    distance_to_combined2 = euclidean_distance(
        inlier_mean, scale_center - pointer_direction
    )

    # If the inliers' mean coordinate is closer to the inversed unit vector
    # that originates from the scale center, inverse the pointer direction.
    if distance_to_combined1 > distance_to_combined2:
        pointer_direction = -pointer_direction

    # Get the delta on each axis for pointer direction unit vector
    delta_x, delta_y = pointer_direction

    # Calculate the angle. The y-axis is inversed because traditional plotting
    # has it's origo at bottom-left and OpenCV has it's "origo" at top-left.
    pointer_angle = np.degrees(np.arctan2(-delta_y, delta_x)) % 360

    # Map the measured angle to a value on scale
    value = map_angle_to_value(
        pointer_angle,
        scale_angle_1,
        scale_angle_2,
        scale_value_1,
        scale_value_2,
        force_limits=False,
        directional=True,
    )

    timestamp_stop = time.time()

    if return_verbose:
        timestamps = [
            ("init", timestamp_start_init),
            ("contour_cleanup", timestamp_start_contour_cleanup),
            ("contour_thinning", timestamp_start_contour_thinning),
            ("line_finding", timestamp_start_line_finding),
            ("pointer_calculation", timestamp_start_pointer_calculation),
            ("end", timestamp_stop),
        ]

        durations = [
            (name, next_timestamp - prev_timestamp)
            for (name, prev_timestamp), (_, next_timestamp) in zip(
                timestamps, timestamps[1:]
            )
        ]

        return (
            value,
            pointer_angle,
            pointer_direction,
            points,
            [found_line],
            durations,
        )
    else:
        return value


# ---------------------- METER READER IMPLEMENTATION 2 ---------------------- #
def implementation_2(detection, mask, return_verbose=False):
    timestamp_start_init = time.time()

    # Extract values
    width, height = mask.shape[1::-1]
    meta = detection.template.meta

    template_width = meta["crop_width"]
    template_height = meta["crop_height"]
    scale_center_x = meta["center_x"] * (width / template_width)
    scale_center_y = meta["center_y"] * (height / template_height)

    scale_center = np.array([scale_center_x, scale_center_y])

    scale_angle_1 = meta["scale_angle_1"]
    scale_value_1 = meta["scale_value_1"]
    scale_angle_2 = meta["scale_angle_2"]
    scale_value_2 = meta["scale_value_2"]

    # Calculate minimum required area for the pointer
    minimum_area = width * height * 0.004

    timestamp_start_edge_finding = time.time()

    # Find points on the contour edges
    points = mask_get_edge_points(mask, minimum_size=minimum_area)

    timestamp_start_line_finding = time.time()

    # Find two lines from the set of points.
    # Distance from each line must be at most 10% of `min(width, height)` of
    # the mask.
    found_lines = list(
        find_n_lines_near_point(
            points,
            2,
            (scale_center_x, scale_center_y),
            min(width, height) * 0.1,
        )
    )

    # If two lines were found, calculate the mean of the two lines
    if len(found_lines) == 2:
        # Get unit vectors of the lines and the RANSAC inliers
        ((_, vector1), inliers1), ((_, vector2), inliers2) = found_lines

        timestamp_start_pointer_calculation = time.time()

        # Get the number of inliers in both lines
        inlier_count = len(inliers1) + len(inliers2)

        # Because the pointer direction unit vectors are non-directional, both
        # directions are considered for both lines. First, the unit vectors are
        # added together to get a vector `a` and subtracted to get a vector
        # `b`. Then the one with larger distance to the orig is considered to
        # be parallel to the pointer (later `x`). However, `x` is still
        # non-directional. The direction is determined by comparing the
        # distance from `x` and `-x` to inliers' mean coordinate. The one with
        # smaller distance is considered to be the directional vector for the
        # pointer.

        # Calculate the mean coordinate for the points that were used to define
        # the two lines by RANSAC
        inlier_mean = (
            inliers1.sum(axis=0) + inliers2.sum(axis=0)
        ) / inlier_count

        # Calculate which vector combination produces the longest vector. It
        # always results in two vectors `v` and `-v`.
        combined1, combined2 = longest_vector_combinations(
            vector1, vector2, normalize=True
        )

        # Calculate the distance from inliers' mean coordinate to the vector
        # `x`.
        distance_to_combined1 = euclidean_distance(
            inlier_mean, scale_center + combined1
        )
        # Calculate the distance from inliers' mean coordinate to the vector
        # `-x`.
        distance_to_combined2 = euclidean_distance(
            inlier_mean, scale_center + combined2
        )

        # Set the vector with smaller distance to `pointer_direction`
        if distance_to_combined1 > distance_to_combined2:
            pointer_direction = combined2
        else:
            pointer_direction = combined1

    # If one line is found, use that direction as-is
    elif len(found_lines) == 1:
        # Get the unit vector that is parallel to the pointer line and the
        # RANSAC inliers
        (_, pointer_direction), inliers = found_lines[0]

        timestamp_start_pointer_calculation = time.time()

        # Calculate the mean coordinate for the points that were used to define
        # the line by RANSAC
        inlier_mean = inliers.mean(axis=0)

        # Calculate the distance from inliers' mean coordinate to the unit
        # vector that originates from the scale center.
        distance_to_combined1 = euclidean_distance(
            inlier_mean, scale_center + pointer_direction
        )
        # Calculate the distance from inliers' mean coordinate to the inverse
        # unit vector that originates from the scale center.
        distance_to_combined2 = euclidean_distance(
            inlier_mean, scale_center - pointer_direction
        )

        # If the inliers' mean coordinate is closer to the inversed unit vector
        # that originates from the scale center, inverse the pointer direction.
        if distance_to_combined1 > distance_to_combined2:
            pointer_direction = -pointer_direction

    else:
        if return_verbose:
            timestamp_stop = time.time()
            timestamps = [
                ("init", timestamp_start_init),
                ("edge_finding", timestamp_start_edge_finding),
                ("line_finding", timestamp_start_line_finding),
                ("end", timestamp_stop),
            ]
            durations = [
                (name, next_timestamp - prev_timestamp)
                for (name, prev_timestamp), (_, next_timestamp) in zip(
                    timestamps, timestamps[1:]
                )
            ]
            return (None, None, None, points, [], durations)
        else:
            return None

    # Get the delta on each axis for pointer direction unit vector
    delta_x, delta_y = pointer_direction

    # Calculate the angle. The y-axis is inversed because traditional plotting
    # has it's origo at bottom-left and OpenCV has it's "origo" at top-left.
    pointer_angle = np.degrees(np.arctan2(-delta_y, delta_x)) % 360

    # Map the measured angle to a value on scale
    value = map_angle_to_value(
        pointer_angle,
        scale_angle_1,
        scale_angle_2,
        scale_value_1,
        scale_value_2,
        force_limits=False,
        directional=True,
    )

    timestamp_stop = time.time()

    if return_verbose:
        timestamps = [
            ("init", timestamp_start_init),
            ("edge_finding", timestamp_start_edge_finding),
            ("line_finding", timestamp_start_line_finding),
            ("pointer_calculation", timestamp_start_pointer_calculation),
            ("end", timestamp_stop),
        ]

        durations = [
            (name, next_timestamp - prev_timestamp)
            for (name, prev_timestamp), (_, next_timestamp) in zip(
                timestamps, timestamps[1:]
            )
        ]

        return (
            value,
            pointer_angle,
            pointer_direction,
            points,
            found_lines,
            durations,
        )
    else:
        return value


# ------------------------ MAIN APPLICATION UTILITIES ----------------------- #
def iterate_images(datasetdir):
    annotationpath = datasetdir / "annotations.yml"

    annotations = yaml.load(annotationpath.read_text(), Loader=Loader)

    for dirname, annotated_values in annotations.items():
        for imagepath in (datasetdir / dirname).glob("*.*"):
            if (
                imagepath.suffix.lower() in {".jpg", ".jpeg", ".png"}
                and imagepath.stem not in annotated_values.keys()
            ):
                yield imagepath, annotated_values


def visualize_reading(
    image_original, detection, points, lines, pointer_direction
):
    if len(image_original.shape) == 2:
        image_original = cv2.cvtColor(image_original, cv2.COLOR_GRAY2BGR)
    else:
        image_original = image_original.copy()

    images = []

    image = image_original.copy()
    for x, y in points:
        cv2.circle(image, (x, y), 1, (0, 0, 255))

    width, height = image.shape[1::-1]
    meta = detection.template.meta
    template_width = meta["crop_width"]
    template_height = meta["crop_height"]
    scale_center_x = meta["center_x"] * (width / template_width)
    scale_center_y = meta["center_y"] * (height / template_height)
    scale_center = np.int64([scale_center_x, scale_center_y])

    for (linepoint, direction), inliers in lines:
        for x, y in inliers:
            cv2.circle(image, (x, y), 1, (0, 255, 0))
    images.append(image)

    image = image_original.copy()
    for (linepoint, direction), inliers in lines:
        cv2.line(
            image,
            (linepoint - direction * 1000).round().astype(np.int64),
            (linepoint + direction * 1000).round().astype(np.int64),
            (255, 0, 255),
            2,
        )
    images.append(image)

    image = image_original.copy()
    if pointer_direction is not None:
        cv2.line(
            image,
            scale_center,
            (scale_center + pointer_direction * 1000).round().astype(np.int64),
            (255, 128, 0),
            2,
        )
    images.append(image)

    return images


# ----------------------------- MAIN APPLICATION ---------------------------- #

# Load template images
logger.info("Loading template images")
meters = {
    path.stem: TemplateImage(path)
    for path in sorted(list(meterannotationdir.glob("*.png")))
}

# Create readers for both implementations
reader1 = MeterReader(implementation_1)
reader2 = MeterReader(implementation_2)

for name, templateimage in meters.items():
    reader1.add_template(name, templateimage)
    reader2.add_template(name, templateimage)

logger.success("Template images loaded")

detectionimage_colors = [
    (255, 0, 0),
    (255, 0, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
]

logger.info("Starting processing")

imagepath_annotations = list(iterate_images(datasetdir))

for imageindex, (imagepath, annotation) in enumerate(imagepath_annotations):
    logger.info(
        "Processing image {}/{}", imageindex + 1, len(imagepath_annotations)
    )
    image = cv2.imread(str(imagepath))

    logger.info("[IMPLEMENTATION-1] Finding template images")
    duration_find_templates_reader1 = -time.time()
    reader1.find_templates(image)
    duration_find_templates_reader1 += time.time()
    logger.success(
        "[IMPLEMENTATION-1] Templates found in {:.3f} seconds",
        duration_find_templates_reader1,
    )

    logger.info("[IMPLEMENTATION-2] Finding template images")
    duration_find_templates_reader2 = -time.time()
    reader2.find_templates(image)
    duration_find_templates_reader2 += time.time()
    logger.success(
        "[IMPLEMENTATION-2] Templates found in {:.3f} seconds",
        duration_find_templates_reader2,
    )

    logger.info("[IMPLEMENTATION-1] Reading meter values")
    duration_detect_reader1 = -time.time()
    readings1 = reader1.detect(
        image, visual_segmentation=False, draw_detection=False
    )
    duration_detect_reader1 += time.time()
    for metername, data in readings1.items():
        logger.info('Processing step durations for meter "{}"', metername)
        for operation, duration in data[6]:
            logger.info(
                '[IMPLEMENTATION-1] [{}] Operation "{}" took {:.3f} seconds',
                metername,
                operation,
                duration,
            )
    logger.success(
        "[IMPLEMENTATION-1] Meter values read in {:.3f} seconds",
        duration_find_templates_reader1,
    )

    logger.info("[IMPLEMENTATION-2] Reading meter values")
    duration_detect_reader2 = -time.time()
    readings2 = reader2.detect(
        image, visual_segmentation=False, draw_detection=False
    )
    duration_detect_reader2 += time.time()
    for metername, data in readings2.items():
        logger.info('Processing step durations for meter "{}"', metername)
        for operation, duration in data[6]:
            logger.info(
                '[IMPLEMENTATION-2] [{}] Operation "{}" took {:.3f} seconds',
                metername,
                operation,
                duration,
            )
    logger.success(
        "[IMPLEMENTATION-2] Meter values read in {:.3f} seconds",
        duration_find_templates_reader2,
    )

    detectionimage = image.copy()

    logger.info("Writing results and visualizations")

    for meterindex, metername in enumerate(sorted(readings1.keys())):
        outputdir = (
            imagepath.parent
            / imagepath.stem
            / "{}-implementation_1".format(metername)
        )
        outputdir.mkdir(parents=True, exist_ok=True)

        value, result, points, lines, direction, images, durations = readings1[
            metername
        ]
        demos = visualize_reading(images[-3], result, points, lines, direction)

        result.draw_detection(
            detectionimage,
            color=detectionimage_colors[meterindex],
            thickness=5,
        )

        meta = reader1.templates[metername].meta
        scale_range = get_full_rotation_value(
            meta["scale_angle_1"],
            meta["scale_angle_2"],
            meta["scale_value_1"],
            meta["scale_value_2"],
        )

        with (outputdir / "values.json").open("w") as f:
            if value is not None and annotation[metername] != "x":
                error_in_value = value - annotation[metername]
                error_in_degrees = (value - annotation[metername]) / (
                    scale_range / 360
                )
            else:
                error_in_value = None
                error_in_degrees = None

            json.dump(
                {
                    "value": value,
                    "annotation": annotation[metername],
                    "full_rotation_value": scale_range,
                    "error_in_value": error_in_value,
                    "error_in_degrees": error_in_degrees,
                    "durations": durations,
                },
                f,
                indent=2,
            )

        transformed = images[1]
        segmented_mask = images[3]
        ransac_points = demos[0]
        found_lines = demos[1]
        pointer_line = demos[2]

        cv2.imwrite(
            str(outputdir / "0_template.png"),
            reader1.templates[metername].image,
        )
        cv2.imwrite(str(outputdir / "1_transformed.png"), transformed)
        cv2.imwrite(str(outputdir / "2_segmented_mask.png"), segmented_mask)
        cv2.imwrite(str(outputdir / "3_ransac_points.png"), ransac_points)
        cv2.imwrite(str(outputdir / "4_found_lines.png"), found_lines)
        cv2.imwrite(str(outputdir / "5_pointer_line.png"), pointer_line)

    for metername in sorted(readings2.keys()):
        outputdir = (
            imagepath.parent
            / imagepath.stem
            / "{}-implementation_2".format(metername)
        )
        outputdir.mkdir(parents=True, exist_ok=True)

        value, result, points, lines, direction, images, durations = readings2[
            metername
        ]
        demos = visualize_reading(images[-3], result, points, lines, direction)

        meta = reader2.templates[metername].meta
        scale_range = get_full_rotation_value(
            meta["scale_angle_1"],
            meta["scale_angle_2"],
            meta["scale_value_1"],
            meta["scale_value_2"],
        )

        with (outputdir / "values.json").open("w") as f:
            if value is not None and annotation[metername] != "x":
                error_in_value = value - annotation[metername]
                error_in_degrees = (value - annotation[metername]) / (
                    scale_range / 360
                )
            else:
                error_in_value = None
                error_in_degrees = None

            json.dump(
                {
                    "value": value,
                    "annotation": annotation[metername],
                    "full_rotation_value": scale_range,
                    "error_in_value": error_in_value,
                    "error_in_degrees": error_in_degrees,
                    "durations": durations,
                },
                f,
                indent=2,
            )

        transformed = images[1]
        segmented_mask = images[3]
        ransac_points = demos[0]
        found_lines = demos[1]
        pointer_line = demos[2]

        cv2.imwrite(
            str(outputdir / "0_template.png"),
            reader2.templates[metername].image,
        )
        cv2.imwrite(str(outputdir / "1_transformed.png"), transformed)
        cv2.imwrite(str(outputdir / "2_segmented_mask.png"), segmented_mask)
        cv2.imwrite(str(outputdir / "3_ransac_points.png"), ransac_points)
        cv2.imwrite(str(outputdir / "4_found_lines.png"), found_lines)
        cv2.imwrite(str(outputdir / "5_pointer_line.png"), pointer_line)

    cv2.imwrite(
        str(imagepath.parent / imagepath.stem / "detection.png"),
        detectionimage,
    )

    logger.success("Processing done")
