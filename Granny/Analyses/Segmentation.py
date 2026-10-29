"""
This module performs instance segmentation on user-provided image files.
The analysis is conducted as follows:
    1. retrieves the machine learning (instance segmentation) trained models
       from https://osf.io/. to the current directory 'Analyses/'. The machine
       learning models are uploaded manually and should be named in this
       convention: granny-v{granny_version}-{model_name}-v{model_version}.pt
    2. parses user's input for image folder, initiates a list of Granny.Models.
       Images.Image, then runs YOLOv8 on the images.
    3. extracts and processes the segmented instances from the images.

date: July 12, 2024
author: Nhan H. Nguyen
"""

import colorsys
import itertools
import os
import pathlib
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib import request

import cv2
import numpy as np
import pandas as pd
from Granny.Analyses.Analysis import Analysis
from Granny.Models.AIModel.AIModel import AIModel
from Granny.Models.AIModel.YoloModel import YoloModel
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.Values.FileNameValue import FileNameValue
from Granny.Models.Values.FloatValue import FloatValue
from Granny.Models.Values.ImageListValue import ImageListValue
from Granny.Models.Values.IntValue import IntValue
from Granny.Models.Values.StringValue import StringValue
from numpy.typing import NDArray


class SegmentationConfig:
    """
    Configuration class for instance segmentation module.

    This class provides the configuration details for different segmentation models,
    including the class labels and model details.

    Attributes:
        CLASSES (Dict[str, int]): A dictionary mapping class names to their respective integer labels.
            Format:
                {
                    "class_name_1": int_label_1,
                    "class_name_2": int_label_2,
                    ...
                }

        MODELS (Dict[str, Dict[str, str]]): A dictionary containing model details. Each model is represented
            by another dictionary that includes the full name of the model file and the URL to download it from.
            format:
                {
                    "model_name_1": {
                        "model_full_name_1": "*.pt",
                        "osf_urf_1": "https://osf_link/download/"
                    }
                }
                {
                    "model_name_2": {
                        "model_full_name_2": "*.pt",
                        "osf_urf_1": "https://osf_link/download/"
                    }
                }
    """

    CLASSES: Dict[str, int] = {"fruits": 0, "tray_info": 1}
    MODELS: Dict[str, Dict[str, str]] = {
        "pome_fruit-v1_0": {
            "full_name": "granny-v1_0-pome_fruit-v1_0.pt",
            "url": "https://osf.io/vyfhm/download/",
        }
    }


def parseTrayLayout(path: str) -> List[List[str]]:
    """
    Reads a tray layout file used by research mode.

    The file looks like the tray does in the image: one line per row of fruit, top row first.
    Each line lists, from left to right as seen in the image, the label of each fruit position
    (cells can be separated by commas, semicolons or tabs, so a CSV saved from a spreadsheet works).
    Blank lines, empty cells and lines starting with '#' are ignored. For example, a tray with
    rows of 4, 5, 4 and 5 fruit, numbered left to right starting at the top row, is written as:

        1,2,3,4
        5,6,7,8,9
        10,11,12,13
        14,15,16,17,18

    Labels made of digits are zero-padded so they sort and read well in file names (9 becomes 09).
    Other labels (such as A1) are used as written, with unusual characters replaced by '_'.

    Args:
        path (str): Path of the layout file.

    Returns:
        List[List[str]]: The file-name-ready labels, one list per row, top row first.

    Raises:
        ValueError: If the file can't be read, has no labels, or uses a label more than once.
    """
    try:
        with open(path, encoding="utf-8-sig") as layout_file:
            lines = layout_file.read().splitlines()
    except OSError as err:
        raise ValueError(f"cannot read the tray layout file '{path}' ({err.strerror or err})")

    raw_rows: List[List[str]] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cells = [cell.strip() for cell in re.split(r"[,;\t]", line)]
        cells = [cell for cell in cells if cell]
        if cells:
            raw_rows.append(cells)
    if not raw_rows:
        raise ValueError(f"the tray layout file '{path}' does not contain any labels")

    digit_widths = [len(c) for row in raw_rows for c in row if re.fullmatch(r"[0-9]+", c)]
    width = max([2] + digit_widths)

    rows: List[List[str]] = []
    seen: Dict[str, str] = {}
    for row in raw_rows:
        labels = []
        for cell in row:
            if re.fullmatch(r"[0-9]+", cell):
                label = str(int(cell)).zfill(width)
            else:
                label = re.sub(r"[^A-Za-z0-9_-]", "_", cell)
            if label in seen:
                raise ValueError(
                    f"the tray layout file '{path}' uses the label '{cell}' more than once "
                    f"(also as '{seen[label]}'); every position needs its own label"
                )
            seen[label] = cell
            labels.append(label)
        rows.append(labels)
    return rows


def parseTrayReference(path: str, layout_rows: List[List[str]]) -> Dict[str, Tuple[float, float]]:
    """
    Reads a tray reference file: where each tray position sits in the picture when the tray is
    the right way up, as a fraction of the image width (x) and height (y).

    Granny writes this file itself (tray_reference.csv, in the results folder) after a research
    run on photos of trays that were numbered without doubt. It has the columns label, x, y (and
    n_images, which is ignored). A reference may lack some positions; those positions simply
    can't be used to place a fruit.

    Raises:
        ValueError: with a plain-language message when the file can't be used.
    """
    try:
        table = pd.read_csv(path, dtype={"label": str}, comment="#")
    except Exception as err:
        raise ValueError(f"could not read the tray reference file '{path}': {err}")
    if not {"label", "x", "y"} <= set(table.columns):
        raise ValueError(
            f"the tray reference file '{path}' must have the columns label, x and y. Use the "
            + "tray_reference.csv that Granny writes in the results folder after a research run."
        )
    known = {label for row in layout_rows for label in row}
    spots: Dict[str, Tuple[float, float]] = {}
    for _, row in table.iterrows():
        label = str(row["label"]).strip()
        if label not in known:
            raise ValueError(
                f"the tray reference lists the position '{label}', which is not in the tray layout "
                + "file. The reference and the layout must describe the same tray."
            )
        try:
            x, y = float(row["x"]), float(row["y"])
        except (TypeError, ValueError):
            raise ValueError(f"the tray reference has a position ('{label}') without numbers for x and y.")
        if label in spots:
            raise ValueError(f"the tray reference lists the position '{label}' twice.")
        spots[label] = (x, y)
    if len(spots) < 3:
        raise ValueError("the tray reference needs at least 3 positions to be useful.")
    return spots


def buildTrayReference(
    samples: List[Dict[str, Tuple[float, float]]]
) -> Dict[str, Tuple[float, float, int]]:
    """
    Combines tray images into one reference: the position of each label, as the median over the
    images. Each image is first shifted to line up with the first one (by the median shift of the
    labels they share), so photos taken with the camera or tray moved a little don't blur it.

    Args:
        samples: One dict per tray image, label -> (x, y) as fractions of the image size.

    Returns:
        label -> (x, y, number of images that contributed).
    """
    if not samples:
        return {}
    first = samples[0]
    aligned: List[Dict[str, Tuple[float, float]]] = []
    for sample in samples:
        common = [label for label in sample if label in first]
        if len(common) < 3:
            continue
        shift = np.median([[sample[l][0] - first[l][0], sample[l][1] - first[l][1]] for l in common], axis=0)
        aligned.append({l: (x - shift[0], y - shift[1]) for l, (x, y) in sample.items()})
    combined: Dict[str, Tuple[float, float, int]] = {}
    for label in {l for sample in aligned for l in sample}:
        points = np.array([sample[label] for sample in aligned if label in sample])
        combined[label] = (float(np.median(points[:, 0])), float(np.median(points[:, 1])), len(points))
    return combined


def findColorCard(image: NDArray[np.uint8]) -> Optional[Tuple[float, float]]:
    """
    Looks for a ColorChecker-style color card (a block of 24 same-sized square patches, 6 x 4)
    and returns the (x, y) pixel position of its center, or None when there is none.

    The card is recognized by its patches rather than by its colors: small, square, similar-sized
    outlines are collected, and a group of at least 14 of them that fills about a 6 x 4 block of
    patches is taken to be the card. Labels, notes, tray pockets and fruit do not form such a
    group. Works whichever way up the card is, and needs only OpenCV's standard functions.

    Args:
        image: The image to search (any channel order).

    Returns:
        The center of the card in the pixel coordinates of `image`, or None if no card was found.
    """
    h, w = image.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    long_side = max(small.shape[:2])
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.dilate(cv2.Canny(gray, 20, 60), np.ones((2, 2), np.uint8))
    outlines, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    squares: List[Tuple[float, float, float]] = []  # (x, y, side)
    for outline in outlines:
        area = cv2.contourArea(outline)
        if area < 0.5 * (0.008 * long_side) ** 2 or area > (0.05 * long_side) ** 2:
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(outline)
        if min(rw, rh) < 1 or max(rw, rh) / min(rw, rh) > 1.35 or area / (rw * rh) < 0.8:
            continue
        squares.append((cx, cy, (rw + rh) / 2))
    if len(squares) < 14:
        return None

    # a patch gives both an outer and an inner outline: keep one of each pair
    patches: List[Tuple[float, float, float]] = []
    for sq in squares:
        if all(np.hypot(sq[0] - o[0], sq[1] - o[1]) > 0.5 * min(sq[2], o[2]) for o in patches):
            patches.append(sq)
    pts = np.array(patches)

    best: Optional[Tuple[int, NDArray[np.float64]]] = None
    for x0, y0, side in pts:
        group = pts[(np.abs(pts[:, 2] - side) < 0.3 * side) & (np.hypot(pts[:, 0] - x0, pts[:, 1] - y0) < 7.0 * side)]
        if not 14 <= len(group) <= 40:
            continue
        center = group[:, :2].mean(0)
        _, _, axes = np.linalg.svd(group[:, :2] - center, full_matrices=False)
        proj = (group[:, :2] - center) @ axes.T
        extent = proj.max(0) - proj.min(0)
        pitch = side * 1.25
        long_ext, short_ext = float(extent.max()), float(max(extent.min(), 1.0))
        if not (3.0 <= long_ext / pitch + 1 <= 7.5 and 2.0 <= short_ext / pitch + 1 <= 5.0):
            continue
        if not 1.1 <= long_ext / short_ext <= 2.4:
            continue
        if best is None or len(group) > best[0]:
            best = (len(group), center)
    if best is None:
        return None
    return float(best[1][0] / scale), float(best[1][1] / scale)


@dataclass
class FruitInstances:
    """
    The fruit found on one tray image, in numbering order.

    This is created once per tray image and used for both the segmented fruit files and the
    clean full masked image, so the two always agree on which fruit has which number.

    Attributes:
        boxes: (N, 6) array of [x1, y1, x2, y2, confidence, class] for the kept fruit.
        masks: (N, H, W) array with one binary mask per kept fruit.
        labels: The number (or tray-layout label) of each kept fruit, as used in the file name.
        placement: How each label was decided: 'sequential' (counted in tray order),
            'layout' (matched to the tray layout), 'inferred' (matched to the tray layout from
            the fruit's position because a fruit is missing from its row), 'reference' (matched
            to the optional tray reference because the rows could not place it), or 'unplaced'
            (research mode could not tell which tray position the fruit is in).
        rejected: (box, reason, mask) for each detection discarded as a false detection.
        report: Research-mode check of this tray image (counts, status, notes).
    """

    boxes: Any
    masks: Any
    labels: List[str]
    placement: List[str]
    rejected: List[Tuple[Any, str, Any]] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)


class Segmentation(Analysis):
    """
    Segmentation class for analyzing and segmenting images to identify fruit.

    This class extends the `Analysis` base class and provides functionality for
    setting up and performing segmentation analysis using specified models. It
    manages input and output directories for images and results, and allows for
    specifying models either by name or file path.

    Attributes:
        __analysis_name__ (str): The name of the analysis type, set to "segmentation".
        config (SegmentationConfig): Configuration settings for segmentation.
        analysis_time (str): The timestamp when the analysis instance is created.
        models (list): List of available models for segmentation.
        model (FileNameValue): Specifies the model for segmentation, with options for known
            model names or file paths. Default is "pome_fruit-v1_0".
        input_images (ImageListValue): Directory path for input images, required for analysis.
        seg_images (ImageListValue): Directory path for saving segmented images, defaults to
            a timestamped folder within "results/segmentation".
        tray_infos (ImageListValue): Directory path for saving tray information of analyzed images,
            defaults to a timestamped folder within "results/segmentation".
        masked_images (ImageListValue): Directory path for saving the unfiltered full-masked images
            (everything the model found, no numbering), defaults to a timestamped folder within
            "results/segmentation".
        masked_images_clean (ImageListValue): Directory path for saving the clean full-masked images
            (numbered, false detections removed).
        seg_images_false_positive (ImageListValue): Directory path for saving the detections that
            were discarded as false detections.
    """

    __analysis_name__ = "segmentation"

    def __init__(self):
        super().__init__()
        self.config = SegmentationConfig
        self.analysis_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

        # set from the tray layout file when running in research mode (see _checkOptions)
        self.layout_rows: Optional[List[List[str]]] = None
        # filled in as each tray image is processed, and written out as csv files at the end
        self._manifest_rows: List[Dict[str, Any]] = []
        self._false_positive_rows: List[Dict[str, Any]] = []
        self._report_rows: List[Dict[str, Any]] = []

        # selects a model from the list to be used in this analysis
        self.models = self.config.MODELS
        self.model = FileNameValue(
            "model",
            "model",
            "Specifies the model that should be used for segmentation to identify fruit. The "
            + "model can be specified using a known model name (e.g. 'pome_fruit-v1_0'), "
            + "and Granny will automatically retrieve the model from the online "
            + "https://osf.io. "
            + "Otherwise the value must be a path to where the model is stored on the local "
            + "file system. If no model is specified then the default model will be used.",
        )
        self.model.setValue("pome_fruit-v1_0")
        self.model.setIsRequired(False)

        self.input_images: ImageListValue = ImageListValue(
            "input", "input", "The directory where input images are located."
        )
        self.input_images.setIsRequired(True)

        # YOLO confidence threshold parameter
        self.conf_threshold = FloatValue(
            "conf",
            "confidence",
            "Confidence threshold for YOLO detections. Only detections with confidence "
            + "scores above this threshold will be kept. Range is 0.0 to 1.0, "
            + "default is 0.25.",
        )
        self.conf_threshold.setMin(0.0)
        self.conf_threshold.setMax(1.0)
        self.conf_threshold.setValue(0.25)
        self.conf_threshold.setIsRequired(False)

        # YOLO IOU threshold parameter
        self.iou_threshold = FloatValue(
            "iou",
            "iou_threshold",
            "Intersection over Union (IOU) threshold for non-maximum suppression. "
            + "Used to filter overlapping detections. Range is 0.0 to 1.0, "
            + "default is 0.45.",
        )
        self.iou_threshold.setMin(0.0)
        self.iou_threshold.setMax(1.0)
        self.iou_threshold.setValue(0.45)
        self.iou_threshold.setIsRequired(False)

        # Visualization parameters
        self.mask_alpha = FloatValue(
            "mask_alpha",
            "mask_alpha",
            "Alpha transparency value for mask overlay on output images. "
            + "Range is 0.0 (transparent) to 1.0 (opaque), default is 0.5.",
        )
        self.mask_alpha.setMin(0.0)
        self.mask_alpha.setMax(1.0)
        self.mask_alpha.setValue(0.5)
        self.mask_alpha.setIsRequired(False)

        self.color_brightness = FloatValue(
            "color_brightness",
            "color_brightness",
            "Brightness value for mask colors in HSV color space. "
            + "Range is 0.0 (dark) to 1.0 (bright), default is 1.0.",
        )
        self.color_brightness.setMin(0.0)
        self.color_brightness.setMax(1.0)
        self.color_brightness.setValue(1.0)
        self.color_brightness.setIsRequired(False)

        self.bbox_thickness = IntValue(
            "bbox_thickness",
            "bbox_thickness",
            "Thickness of bounding box lines in pixels. Default is 5.",
        )
        self.bbox_thickness.setMin(1)
        self.bbox_thickness.setMax(50)
        self.bbox_thickness.setValue(5)
        self.bbox_thickness.setIsRequired(False)

        self.font_scale = FloatValue(
            "font_scale",
            "font_scale",
            "Font scale for text labels on output images. Default is 2.0.",
        )
        self.font_scale.setMin(0.1)
        self.font_scale.setMax(10.0)
        self.font_scale.setValue(2.0)
        self.font_scale.setIsRequired(False)

        self.text_thickness = IntValue(
            "text_thickness",
            "text_thickness",
            "Thickness of text labels in pixels. Default is 3.",
        )
        self.text_thickness.setMin(1)
        self.text_thickness.setMax(50)
        self.text_thickness.setValue(3)
        self.text_thickness.setIsRequired(False)

        # Sorting/grouping parameter
        self.row_tolerance = IntValue(
            "row_tolerance",
            "row_tolerance",
            "Row grouping tolerance factor. Fruits are grouped into rows when their y-centers "
            + "differ by more than height/row_tolerance pixels. Smaller values = looser grouping, "
            + "larger values = tighter grouping. Default is 20.",
        )
        self.row_tolerance.setMin(1)
        self.row_tolerance.setMax(100)
        self.row_tolerance.setValue(20)
        self.row_tolerance.setIsRequired(False)

        # False-detection filter parameter
        self.min_area_ratio = FloatValue(
            "min_area_ratio",
            "min_area_ratio",
            "Fruit detections whose bounding-box area is smaller than this fraction of the "
            + "median fruit detection area on the same tray image are discarded as false "
            + "detections (e.g. a sliver of tray rim at the edge of the image). "
            + "Range is 0.0 to 1.0. Set to 0 to disable the filter. Default is 0.3.",
        )
        self.min_area_ratio.setMin(0.0)
        self.min_area_ratio.setMax(1.0)
        self.min_area_ratio.setValue(0.3)
        self.min_area_ratio.setIsRequired(False)

        self.max_aspect_ratio = FloatValue(
            "max_aspect_ratio",
            "max_aspect_ratio",
            "Fruit detections whose bounding box is longer than this many times its shorter "
            + "side are discarded as false detections (e.g. a long strip of tray or "
            + "background). A round fruit is close to 1, and elongated fruit such as pears "
            + "are usually below 2. Set to 0 to disable the filter. Default is 3.0.",
        )
        self.max_aspect_ratio.setMin(0.0)
        self.max_aspect_ratio.setMax(100.0)
        self.max_aspect_ratio.setValue(3.0)
        self.max_aspect_ratio.setIsRequired(False)

        # Image orientation parameter
        self.orientation = StringValue(
            "orientation",
            "orientation",
            "How each image is turned before fruit are found and numbered. 'auto' (default) "
            + "turns portrait images 90 degrees clockwise so every image is landscape; in "
            + "research mode it then also tries turning each image 0, 90, 180 and 270 degrees "
            + "and keeps the turn that fits the tray layout, so photos taken in different "
            + "orientations are labeled the same way. "
            + "0, 90, 180 or 270 turns every image by that many degrees clockwise instead, "
            + "which keeps the numbering the same when trays are photographed in different "
            + "orientations. Fruit numbers and tray layouts always refer to the image as "
            + "shown in the clean full masked image (full_masked_images_clean).",
        )
        self.orientation.setValidValues(["auto", "0", "90", "180", "270"])
        self.orientation.setValue("auto")
        self.orientation.setIsRequired(False)

        # Where the color card should end up (only used by research mode, see _chooseRotation)
        self.card_position = StringValue(
            "card_position",
            "card_position",
            "Research mode only. Some trays look the same upside down, so the tray layout alone "
            + "cannot tell which way up the photo should be. For those trays Granny looks for a "
            + "color card (ColorChecker) in the photo and turns the image so the card is on this "
            + "side of the tray: top-right (default), top-left, bottom-right or bottom-left. If "
            + "no color card is found the image is not turned and a note is added to the report.",
        )
        self.card_position.setValidValues(["top-right", "top-left", "bottom-right", "bottom-left"])
        self.card_position.setValue("top-right")
        self.card_position.setIsRequired(False)

        # Research mode parameters
        self.mode = StringValue(
            "mode",
            "mode",
            "'standard' (default) numbers the fruit in each image 1, 2, 3 and so on in the "
            + "order Granny finds them (top row first, right to left), for any number of "
            + "fruit. 'research' gives each fruit the label of its position on a tray you "
            + "describe with --tray_layout, checks the number of fruit found against "
            + "--nexpect, and reports any tray image that needs attention. Research mode "
            + "requires --nexpect and --tray_layout.",
        )
        self.mode.setValidValues(["standard", "research"])
        self.mode.setValue("standard")
        self.mode.setIsRequired(False)

        self.nexpect = IntValue(
            "nexpect",
            "nexpect",
            "Research mode only. The number of fruit expected on each tray, for example 18. "
            + "It must equal the number of positions in the tray layout file. Fewer fruit "
            + "than this is not an error (fruit may have been removed), but the tray image "
            + "is flagged in the console, on the clean full masked image and in numbering_report.csv.",
        )
        self.nexpect.setMin(0)
        self.nexpect.setValue(0)
        self.nexpect.setIsRequired(False)

        self.tray_layout = FileNameValue(
            "tray_layout",
            "tray_layout",
            "Research mode only. Path to a text or CSV file that describes the tray: one line "
            + "per row of fruit, top row first, listing the label of each fruit position from "
            + "left to right as the tray looks in the clean full masked image. For a tray with rows "
            + "of 4, 5, 4 and 5 fruit numbered left to right from the top row the file contains "
            + "the four lines 1,2,3,4 then 5,6,7,8,9 then 10,11,12,13 then 14,15,16,17,18.",
        )
        self.tray_layout.setValue("")
        self.tray_layout.setIsRequired(False)

        self.tray_reference = FileNameValue(
            "tray_reference",
            "tray_reference",
            "Research mode only, optional. Path to a tray_reference.csv that Granny wrote in the "
            + "results folder after a run on photos of full trays. It records where each tray "
            + "position sits in the picture, so fruit can still be given the right label when "
            + "many fruit are missing and no row of the tray is complete. It works when the camera "
            + "and tray are in about the same place in every photo: it is only used when the tray has "
            + "moved by less than a third of the distance between neighboring positions, and a "
            + "photo whose tray sits noticeably off the reference is reported for checking. "
            + "A fruit that does not clearly sit on a recorded position is never labeled from it.",
        )
        self.tray_reference.setValue("")
        self.tray_reference.setIsRequired(False)

        self.seg_images = ImageListValue(
            "seg_img",
            "segmented_images",
            "The output directory where analysis' images are written.",
        )
        self.seg_images.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                self.analysis_time,
                "segmented_images",
            )
        )
        self.tray_infos = ImageListValue(
            "info",
            "tray_info",
            "The output directory where tray information of the analysis' images are written.",
        )
        self.tray_infos.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                self.analysis_time,
                "tray_infos",
            )
        )
        self.masked_images = ImageListValue(
            "f_img",
            "full_masked_image",
            "The output directory where the full-masked images are written. These show everything "
            + "the model found on the tray (fruit and false detections alike), with no numbering.",
        )
        self.masked_images.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                self.analysis_time,
                "full_masked_images",
            )
        )
        self.masked_images_clean = ImageListValue(
            "f_img_clean",
            "full_masked_image_clean",
            "The output directory where the clean full-masked images are written: turned as "
            + "specified, false detections removed, every fruit numbered as in its segmented file name.",
        )
        self.masked_images_clean.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                self.analysis_time,
                "full_masked_images_clean",
            )
        )
        self.seg_images_false_positive = ImageListValue(
            "seg_fp",
            "segmented_image_false_positive",
            "The output directory where the detections discarded as false detections are written.",
        )
        self.seg_images_false_positive.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                self.analysis_time,
                "segmented_images_false_positive",
            )
        )

        self.addInParam(
            self.model,
            self.input_images,
            self.conf_threshold,
            self.iou_threshold,
            self.mask_alpha,
            self.color_brightness,
            self.bbox_thickness,
            self.font_scale,
            self.text_thickness,
            self.row_tolerance,
            self.min_area_ratio,
            self.max_aspect_ratio,
            self.orientation,
            self.card_position,
            self.mode,
            self.nexpect,
            self.tray_layout,
            self.tray_reference,
        )

    def _getModelUrl(self, model_name: str):
        """
        Retrieves the download URL for a given model name from the configuration.

        Args:
            model_name (str): The key name of the model in self.models.

        Returns:
        str: Download URL if found, else an empty string.
        """
        model_url = ""
        try:
            model_url = self.models[model_name]["url"]
        except KeyError:
            print(f"Key '{model_name}' not found in configuration.")
        return model_url

    def _downloadTrainedWeights(self, model_url: str, verbose: int = 1):
        """
        Download YOLO8 trained weights from Granny GitHub repository:
        https://github.com/SystemsGenetics/granny/tree/dev-MVC/Granny/Analyses/config/segmentation/

        @param local_model_path: local path of the trained weights
        """
        if verbose > 0:
            print(f"Downloading pretrained model to {self.local_model_path} ...")
        request.urlretrieve(model_url, self.local_model_path)
        if verbose > 0:
            print("... done downloading pretrained model!")

    def _segmentInstances(self, image: NDArray[np.uint8]) -> List[Any]:
        """
        Uses instance segmentation model to predict instances in the image. Instances could be
        tray_info, apples, pears, cross-sections, etc.

        1. Loads the Segmentation model (Yolov8 Segmentation)
        2. Performs instance segmentaion to find fruit and tray information
        3. Returns a list of instance of ultralytics.engine.results.Results

        @param image_instance: An GRANNY.Models.Images.Image instance

        @return resuts: List of ultralytics.engine.results.Results of segmentation results,
        including: masks, boxes, xyxy's, classes, confident scores
        """
        # detects instances on the image

        results = self.segmentation_model.predict(
            image,
            retina_masks=True,
            conf=self.conf_threshold.getValue(),
            iou=self.iou_threshold.getValue()
        )  # type: ignore

        return results

    def _checkOptions(self) -> None:
        """
        Checks the options the user gave, before any image is processed.

        In research mode this also reads the tray layout file. Problems are reported as a plain
        error message (without a traceback) so a mistyped option is easy to spot.
        """

        def fail(message: str) -> None:
            sys.exit(f"error: {message}")

        # Values that fail validation are stored as None, so a mistyped choice shows up here.
        mode = self.mode.getValue()
        if mode not in ("standard", "research"):
            fail("--mode must be 'standard' or 'research'.")
        if self.orientation.getValue() not in ("auto", "0", "90", "180", "270"):
            fail("--orientation must be one of: auto, 0, 90, 180, 270.")
        if self.card_position.getValue() not in ("top-right", "top-left", "bottom-right", "bottom-left"):
            fail("--card_position must be one of: top-right, top-left, bottom-right, bottom-left.")

        self.layout_rows = None
        self.reference_spots = None
        if mode != "research":
            if self.tray_reference.getValue():
                fail("--tray_reference is only used in research mode (add --mode research).")
            return

        nexpect = self.nexpect.getValue()
        if not isinstance(nexpect, int) or nexpect < 1:
            fail(
                "research mode needs --nexpect, the number of fruit you expect on each tray "
                + "(for example --nexpect 18)."
            )
        layout_path = self.tray_layout.getValue()
        if not layout_path:
            fail(
                "research mode needs --tray_layout, the path to a file that describes the "
                + "tray (run with --help to see how to write one)."
            )
        try:
            rows = parseTrayLayout(layout_path)
        except ValueError as err:
            fail(str(err))
        spots = sum(len(row) for row in rows)
        if spots != nexpect:
            fail(
                f"--nexpect is {nexpect} but the tray layout file lists {spots} fruit "
                + "positions. The two must match."
            )
        self.layout_rows = rows

        reference_path = self.tray_reference.getValue()
        if reference_path:
            try:
                self.reference_spots = parseTrayReference(reference_path, rows)
            except ValueError as err:
                fail(str(err))

    def _orientImage(self, image_instance: Image) -> int:
        """
        Turns an image as requested by the orientation option.

        'auto' keeps Granny's original behavior (portrait images are turned 90 degrees clockwise
        so they are landscape); in research mode the tray layout is then used to pick the final
        turn (see `_chooseRotation`). Otherwise every image is turned by the given number of
        degrees clockwise, so trays photographed in different orientations are numbered the same
        way.

        Returns:
            The number of degrees clockwise the image was turned.
        """
        setting = self.orientation.getValue()
        if setting == "auto":
            h, w, _ = image_instance.getShape()
            if h > w:
                image_instance.rotateImage()
                return 90
            return 0
        code = {
            "90": cv2.ROTATE_90_CLOCKWISE,
            "180": cv2.ROTATE_180,
            "270": cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(setting)
        if code is not None:
            image_instance.setImage(cv2.rotate(image_instance.getImage(), code))
        return int(setting)

    @staticmethod
    def _rotateBoxes(
        data: NDArray[np.float32], height: int, width: int, turn: int
    ) -> Tuple[NDArray[np.float32], Tuple[int, int]]:
        """
        Where the boxes of an image of the given size would be if the image were turned.

        Args:
            data: (N, 6) array of [x1, y1, x2, y2, confidence, class].
            height, width: Size of the image the boxes belong to.
            turn: Degrees clockwise (0, 90, 180 or 270).

        Returns:
            The turned boxes, and the (height, width) of the turned image.
        """
        x1, y1, x2, y2 = (data[:, i].astype(np.float64) for i in range(4))
        if turn == 90:
            nx1, nx2, ny1, ny2, shape = height - y2, height - y1, x1, x2, (width, height)
        elif turn == 180:
            nx1, nx2, ny1, ny2, shape = width - x2, width - x1, height - y2, height - y1, (height, width)
        elif turn == 270:
            nx1, nx2, ny1, ny2, shape = y1, y2, width - x2, width - x1, (width, height)
        else:
            return data, (height, width)
        turned = data.copy()
        turned[:, 0], turned[:, 1], turned[:, 2], turned[:, 3] = nx1, ny1, nx2, ny2
        return turned, shape

    @staticmethod
    def _turnPoint(x: float, y: float, height: int, width: int, turn: int) -> Tuple[float, float]:
        """Where pixel (x, y) of a height x width image lands when the image is turned `turn` degrees clockwise."""
        if turn == 90:
            return height - y, x
        if turn == 180:
            return width - x, height - y
        if turn == 270:
            return y, width - x
        return x, y

    def _chooseRotation(self, image_instance: Image) -> Tuple[int, str, bool]:
        """
        Research mode: works out which way up a tray image should be, using the tray layout.

        Each of the four turns (0, 90, 180, 270 degrees clockwise) is tried on the fruit already
        found: the fruit are grouped into rows and matched to the layout exactly as they will be
        for the real numbering, and the turn that lets the most fruit be placed wins. A wrong turn
        gives rows of the wrong sizes, so most fruit stay unplaced; a stray detection or a missing
        fruit only affects a few fruit and doesn't change the winner. The fruit are only found once
        (the caller finds them again only if the image needs turning).

        Some trays look the same upside down (the rows of the layout read the same backwards, like
        a 5 x 5 tray). The fruit cannot say which way up such a photo is, so the position of the
        color card decides: the image is turned so the card sits on the side of the tray given by
        --card_position. If there is no color card, or its position is not clear, the image is
        not turned at all and the note says so, because any choice would be a guess.

        Args:
            image_instance: The image with its segmentation results set, not yet turned.

        Returns:
            The turn to apply (degrees clockwise), a note for the report ("" when there is
            nothing to say), and whether the orientation is certain.
        """
        fruit_cls = SegmentationConfig.CLASSES["fruits"]
        [results] = image_instance.getSegmentationResults()
        if results.boxes is None or results.masks is None:
            return 0, "", True
        boxes = results.boxes.cpu().numpy()
        data = boxes.data
        fruit_idx = np.where(boxes.cls == fruit_cls)[0]
        if len(fruit_idx) == 0:
            return 0, "", True
        keep, _ = self._rejectFalseDetections(data, fruit_idx)
        if len(keep) < 2:
            return 0, "", True
        height, width = boxes.orig_shape

        def score_turns(use_reference: bool, limits: Tuple[float, float] = (0.3, 0.3)) -> Dict[int, Tuple[int, int]]:
            result: Dict[int, Tuple[int, int]] = {}
            for turn in (0, 90, 180, 270):
                turned, shape = self._rotateBoxes(data[keep], height, width, turn)
                df = self._sortInstances(turned, shape)
                _, _, placement, _ = self._assignLayoutLabels(df, shape, use_reference, limits)
                result[turn] = (
                    sum(1 for p in placement if p != "unplaced"),
                    placement.count("layout"),
                )
            return result

        # the rows of the layout decide first; the tray reference (if any) is only consulted when
        # the rows cannot place a single fruit in any turn, and then it must win clearly, because
        # a tray with many fruit missing can look alike in two turns
        scores = score_turns(False)
        best = max(scores.values())
        if best[0] == 0 and self.reference_spots:
            # a tray staggered like this looks nearly the same turned 180 degrees and moved half a
            # position, so here the tray may only have moved very little and the fruit must sit
            # close to their positions (a fixed camera)
            scores = score_turns(True, (0.15, 0.2))
            ranked = sorted(scores.values(), reverse=True)
            best = ranked[0]
            if best[0] >= 3 and best[0] - ranked[1][0] < max(2, 0.25 * best[0]):
                return 0, "the tray reference fits more than one turn about equally well; image not turned", False
            if best[0] >= 3:
                winner = [turn for turn, score in scores.items() if score == best][0]
                return winner, "turned using the tray reference (the rows of the layout could not decide)", True
            return 0, "no turn of the image fits the tray layout or the tray reference", False
        if best[0] == 0:
            return 0, "no turn of the image fits the tray layout", False
        winners = [turn for turn, score in scores.items() if score == best]

        # a layout whose rows read the same from the other end fits equally well turned 180 degrees
        sizes = [len(row) for row in (self.layout_rows or [])]
        symmetric = sizes == sizes[::-1]
        candidates = sorted({w for w in winners} | ({(w + 180) % 360 for w in winners} if symmetric else set()))
        if len(candidates) == 1:
            return candidates[0], "", True

        # the tray looks the same in several turns: only the color card can say which is right
        names = ", ".join(str(t) for t in candidates)
        position = self.card_position.getValue()
        card = findColorCard(image_instance.getImage())
        if card is None:
            return 0, (
                f"the tray fits the layout equally well turned {names} degrees and no color card "
                + "was found to tell them apart; image not turned"
            ), False

        centers = np.column_stack([(data[keep, 0] + data[keep, 2]) / 2, (data[keep, 1] + data[keep, 3]) / 2])
        fruit_center = centers.mean(axis=0)
        wanted = np.array([1.0 if "right" in position else -1.0, -1.0 if position.startswith("top") else 1.0])
        wanted /= np.linalg.norm(wanted)
        fits = {}
        for turn in candidates:
            cx, cy = self._turnPoint(card[0], card[1], height, width, turn)
            fx, fy = self._turnPoint(fruit_center[0], fruit_center[1], height, width, turn)
            direction = np.array([cx - fx, cy - fy])
            fits[turn] = float(direction @ wanted / max(np.linalg.norm(direction), 1e-9))
        ranked = sorted(fits.items(), key=lambda item: -item[1])
        (top_turn, top_fit), (_, second_fit) = ranked[0], ranked[1]
        if top_fit < 0.4 or top_fit - second_fit < 0.5:
            return 0, (
                f"the tray fits the layout equally well turned {names} degrees and the color card "
                + f"is not clearly on the {position} side of the tray; image not turned"
            ), False
        return top_turn, (
            f"the tray fits the layout equally well turned {names} degrees; turned {top_turn} "
            + f"degrees so the color card is on the {position} side"
        ), True

    def _rejectFalseDetections(
        self, data: NDArray[np.float32], indices: NDArray[np.int64]
    ) -> Tuple[NDArray[np.int64], List[Tuple[Any, str, int]]]:
        """
        Splits fruit detections into the ones to keep and the ones discarded as false detections.

        A detection is discarded when its bounding box is unusually small compared with the other
        fruit on the tray (area under min_area_ratio times the median area), or far too elongated
        to be a fruit (longer than max_aspect_ratio times its shorter side). A sliver of tray rim
        at the image edge or a long strip of background are typical examples. With fewer than three
        fruit there is nothing to compare against, so nothing is discarded.

        Args:
            data: (N, 6) array of [x1, y1, x2, y2, confidence, class] for all detections.
            indices: Positions in `data` of the fruit-class detections to check.

        Returns:
            The positions in `data` to keep, and a (detection, reason, position in `data`) triple for
            each one discarded.
        """
        rejected: List[Tuple[Any, str, int]] = []
        if len(indices) < 3:
            return indices, rejected

        min_ratio = self.min_area_ratio.getValue()
        max_aspect = self.max_aspect_ratio.getValue()
        widths = data[indices, 2] - data[indices, 0]
        heights = data[indices, 3] - data[indices, 1]
        areas = widths * heights
        longer = np.maximum(widths, heights)
        shorter = np.maximum(np.minimum(widths, heights), 1)
        none = np.zeros(len(indices), dtype=bool)
        too_small = (areas < min_ratio * np.median(areas)) if min_ratio > 0 else none
        too_long = (longer / shorter > max_aspect) if max_aspect > 0 else none

        for j in np.where(too_small | too_long)[0]:
            if too_small[j]:
                reason = f"area is under {min_ratio:g} x the median fruit area"
            else:
                reason = f"shape is {longer[j] / shorter[j]:.1f}:1, over the {max_aspect:g}:1 limit"
            rejected.append((data[indices[j]], reason, int(indices[j])))
        return indices[~(too_small | too_long)], rejected

    @staticmethod
    def _snapToSlots(xs: NDArray[np.float64], references: List[NDArray[np.float64]]) -> Optional[List[int]]:
        """
        Works out which positions in a row hold the fruit when the row has fewer fruit than spots.

        The x positions of the row's spots are taken from the complete rows of the same size on
        the same image (each one is tried separately, because in a staggered tray the rows sit at
        slightly different x positions). Every way of putting the fruit, in order, into the spots
        is tried against every one of those rows, and the one where the fruit sit closest to the
        spots wins.

        Args:
            xs: x-centers of the fruit found in the row, left to right.
            references: x-centers of all the spots in each complete row of that size, left to
                right (one array per complete row).

        Returns:
            The index of the spot each fruit sits in, or None when the answer is not clear (the
            fruit don't sit on the spots, or two different placements fit about equally well).
        """
        found = len(xs)
        spots = len(references[0])
        if spots < 2:
            return None
        pitch = float(np.mean([np.median(np.diff(ref)) for ref in references]))
        if pitch <= 0:
            return None
        cost_of: Dict[Tuple[int, ...], float] = {}
        for ref in references:
            for combo in itertools.combinations(range(spots), found):
                cost = float(np.abs(xs - ref[list(combo)]).sum())
                cost_of[combo] = min(cost, cost_of.get(combo, cost))
        costs = sorted((cost, combo) for combo, cost in cost_of.items())
        best_cost, best = costs[0]
        if best_cost / found > 0.35 * pitch:
            return None
        if len(costs) > 1 and costs[1][0] - best_cost < 0.5 * pitch:
            return None
        return list(best)

    def _placeByReference(
        self,
        df: pd.DataFrame,
        shape: Tuple[int, int],
        placed: List[Tuple[int, str, str]],
        unplaced: List[int],
        max_shift_factor: float = 0.3,
        tol_factor: float = 0.3,
    ) -> Tuple[int, float]:
        """
        Gives labels to fruit the rows could not place, using the tray reference (optional).

        The reference says where each tray position sits in the picture. The whole tray may have
        moved a little between photos, so first the shift that makes most fruit sit on a reference
        position is found (at least 70 percent of the fruit, and at least 5, must agree, or nothing is
        done; the shift may be at most `max_shift_factor` of the distance between neighboring
        positions). Then each unplaced fruit takes the label of the reference position it sits on, but
        only when it sits clearly closer to that position than to any other (within 0.3 of the
        distance between neighboring positions), the position is not already taken, and no other
        fruit claims it. Everything else stays unplaced.

        Args:
            df: The fruit of the image as returned by `_sortInstances`.
            shape: (height, width) of the image, in pixels.
            placed: Fruit already placed, as (fruit, label, how); extended in place.
            unplaced: Fruit not yet placed; fruit that get a label are removed in place.
            max_shift_factor: The most the tray may have moved, as a fraction of the distance
                between neighboring reference positions.
            tol_factor: How close a fruit must sit to a reference position (after the shift), as
                a fraction of the same distance.

        Returns:
            The number of fruit placed with the reference, and how far the tray was found to sit
            from the reference (as a fraction of the distance between neighboring positions).
        """
        spots = self.reference_spots
        if not spots or not unplaced:
            return 0, 0.0
        height, width = shape
        labels = list(spots)
        ref = np.array([[spots[l][0] * width, spots[l][1] * height] for l in labels])
        spacing = np.linalg.norm(ref[:, None] - ref[None], axis=2)
        np.fill_diagonal(spacing, np.inf)
        pitch = float(np.median(spacing.min(axis=1)))
        tol = tol_factor * pitch

        centers = df[["xcenter", "ycenter"]].to_numpy(dtype=float)
        index_of = {int(n): i for i, n in enumerate(df["nums"])}

        def nearest(shift: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
            dist = np.linalg.norm((centers - shift)[:, None] - ref[None], axis=2)
            return dist.min(axis=1), dist.argmin(axis=1)

        # The shift of the tray: the one that puts the most fruit on reference positions. It may be
        # at most half the distance between neighboring positions: a tray with many fruit missing
        # can look just like a different part of the tray after a larger shift, so a larger shift
        # could not be told apart from a wrong match.
        max_shift = max_shift_factor * pitch
        best_shift, best_score = None, (0, 0.0)
        for shift in np.vstack([np.zeros((1, 2)), (centers[:, None, :] - ref[None, :, :]).reshape(-1, 2)]):
            if np.hypot(shift[0], shift[1]) > max_shift:
                continue
            near, _ = nearest(shift)
            inside = near <= tol
            score = (int(inside.sum()), -float(near[inside].sum()))
            if best_shift is None or score > best_score:
                best_shift, best_score = shift, score
        if best_shift is None:
            return 0, 0.0
        near, which = nearest(best_shift)
        inside = near <= tol
        if inside.sum() < max(5, int(np.ceil(0.7 * len(centers)))):
            return 0, 0.0
        best_shift = best_shift + np.median((centers - ref[which])[inside] - best_shift, axis=0)

        taken = {label for _, label, _ in placed}
        free = [k for k, label in enumerate(labels) if label not in taken]
        if not free:
            return 0, 0.0
        claims: Dict[str, List[int]] = {}
        for fruit in unplaced:
            point = centers[index_of[fruit]] - best_shift
            dist = np.linalg.norm(ref[free] - point, axis=1)
            order = np.argsort(dist)
            if dist[order[0]] > tol:
                continue
            if len(order) > 1 and dist[order[1]] < 1.6 * dist[order[0]]:
                continue
            claims.setdefault(labels[free[order[0]]], []).append(fruit)
        count = 0
        for label, fruits in claims.items():
            if len(fruits) == 1:
                placed.append((fruits[0], label, "reference"))
                unplaced.remove(fruits[0])
                count += 1
        return count, float(np.hypot(best_shift[0], best_shift[1]) / pitch)

    def _assignLayoutLabels(
        self,
        df: pd.DataFrame,
        shape: Tuple[int, int],
        use_reference: bool = True,
        reference_limits: Tuple[float, float] = (0.3, 0.3),
    ) -> Tuple[List[int], List[str], List[str], Dict[str, Any]]:
        """
        Research mode: gives each fruit the label of its position in the tray layout file.

        The fruit are grouped into the rows of the image (top row first) and ordered left to right
        within each row. When the rows match the layout, each fruit takes the label at the same spot
        of the layout. When a row has fewer fruit than spots (fruit removed from the tray), the
        fruit are matched to spots by where they sit, using the spot positions of a complete row
        of the same size on the same image. Fruit that can't be placed with confidence (the number
        of rows differs, a row has too many fruit, or the answer is not clear) are labeled U01, U02
        and so on, so they can never be mistaken for a position.

        If a tray reference was given, fruit the rows could not place are then tried against it
        (see `_placeByReference`).

        Args:
            df: The fruit of one tray image as returned by `_sortInstances`.
            shape: (height, width) of the image, in pixels.
            use_reference: Set to False to leave the tray reference out (used when choosing the
                orientation, where the rows of the layout must have the first say).
            reference_limits: (largest shift, largest distance of a fruit from a position) that
                the tray reference may be trusted with (see `_placeByReference`).

        Returns:
            The fruit indices in label order, their labels, how each label was decided, and a
            report with the counts, a status ('ok', 'fewer', 'more' or 'check') and notes.
        """
        layout = self.layout_rows or []
        nexpect = int(self.nexpect.getValue())
        detected = len(df)
        rows_found = [
            df[df["rows"] == r].sort_values("xcenter").reset_index(drop=True)
            for r in range(1, int(df["rows"].max()) + 1)
        ]

        placed: List[Tuple[int, str, str]] = []
        unplaced: List[int] = []
        missing: List[str] = []

        if len(rows_found) != len(layout):
            for row in rows_found:
                unplaced.extend(int(n) for n in row["nums"])
        else:
            # x positions of the spots in the rows that have all their fruit, by row size
            reference: Dict[int, List[Any]] = {}
            for row, spots in zip(rows_found, layout):
                if len(row) == len(spots):
                    reference.setdefault(len(spots), []).append(row["xcenter"].to_numpy(dtype=float))

            for row, spots in zip(rows_found, layout):
                slots: Optional[List[int]] = None
                how = "layout"
                if len(row) == len(spots):
                    slots = list(range(len(row)))
                elif len(row) < len(spots) and len(spots) in reference:
                    slots = self._snapToSlots(row["xcenter"].to_numpy(dtype=float), reference[len(spots)])
                    how = "inferred"
                if slots is None:
                    unplaced.extend(int(n) for n in row["nums"])
                    continue
                for fruit, slot in zip(row["nums"], slots):
                    placed.append((int(fruit), spots[slot], how))
                missing.extend(spots[s] for s in range(len(spots)) if s not in slots)

        from_reference, reference_shift = (
            self._placeByReference(df, shape, placed, unplaced, *reference_limits) if use_reference else (0, 0.0)
        )
        if from_reference and not unplaced:
            # every fruit has a place now, so the empty positions are the labels nobody took
            used = {label for _, label, _ in placed}
            missing = [label for row in layout for label in row if label not in used]

        order = [fruit for fruit, _, _ in placed]
        labels = [label for _, label, _ in placed]
        placement = [how for _, _, how in placed]
        taken = {label for row in layout for label in row}
        for count, fruit in enumerate(unplaced, start=1):
            label = f"U{count:02d}"
            while label in taken:
                label += "x"
            order.append(fruit)
            labels.append(label)
            placement.append("unplaced")

        # the reference is trusted less when the tray sits far from where it has it
        far_from_reference = from_reference > 0 and reference_shift > 0.15
        if detected > nexpect:
            status = "more"
        elif unplaced or far_from_reference:
            status = "check"
        elif detected < nexpect:
            status = "fewer"
        else:
            status = "ok"

        notes: List[str] = []
        if detected != nexpect:
            notes.append(f"found {detected} fruit, expected {nexpect}")
        if len(rows_found) != len(layout):
            notes.append(f"found {len(rows_found)} rows of fruit, the layout has {len(layout)}")
        if missing:
            notes.append("empty positions: " + ", ".join(sorted(missing)))
        if "inferred" in placement:
            notes.append("some positions were worked out from where the fruit sit")
        if from_reference:
            notes.append(f"{from_reference} fruit were placed using the tray reference")
        if far_from_reference:
            notes.append(
                f"the tray sits {reference_shift:.2f} position spacings from where the tray reference "
                + "has it, so check the labels that came from the reference"
            )
        if unplaced:
            notes.append(f"{len(unplaced)} fruit could not be matched to a tray position (labeled U01, U02, ...)")

        report = {
            "detected": detected,
            "expected": nexpect,
            "status": status,
            "missing": sorted(missing),
            "unplaced": len(unplaced),
            "notes": "; ".join(notes),
            "banner": "" if status == "ok" else "CHECK: " + "; ".join(notes),
        }
        return order, labels, placement, report

    def _getFruitInstances(self, tray_image: Image) -> Optional[FruitInstances]:
        """
        Returns the fruit found on a tray image, with their numbers, in numbering order.

        This is the single source of truth for fruit numbering. The segmented fruit files
        (``..._fruit_NN.png``) and the full masked image both use what this returns, so the number
        drawn on the masked image is always the NN in the matching file name.

        Steps:
            1. keeps only detections of the "fruits" class,
            2. discards false detections (see `_rejectFalseDetections`),
            3. sorts the remaining fruit in tray order (see `_sortInstances`),
            4. numbers them 01, 02, ... in that order (standard mode), or gives each the label of
               its position in the tray layout file and checks the count (research mode).

        Args:
            tray_image (Image): The tray image with segmentation results already set.

        Returns:
            The kept fruit, or None if no fruit was kept.
        """
        fruit_cls = SegmentationConfig.CLASSES["fruits"]
        [results] = tray_image.getSegmentationResults()
        if results.boxes is None or results.masks is None:
            return None
        boxes = results.boxes.cpu().numpy()
        masks = results.masks.cpu().numpy()
        data = boxes.data  # (N, 6): x1, y1, x2, y2, conf, cls
        mask_data = masks.data  # (N, H, W)

        fruit_idx = np.where(boxes.cls == fruit_cls)[0]
        if len(fruit_idx) == 0:
            return None
        keep, rejected = self._rejectFalseDetections(data, fruit_idx)
        for box, reason, _ in rejected:
            print(
                f"{tray_image.getImageName()}: discarded false detection at "
                f"({int(box[0])}, {int(box[1])})-({int(box[2])}, {int(box[3])}), "
                f"confidence {box[4]:.3f}: {reason}"
            )
        if len(keep) == 0:
            return None

        # sorts the kept fruit based on xy-coordinates
        sorted_df = self._sortInstances(data[keep], boxes.orig_shape)

        report: Dict[str, Any] = {}
        if self.layout_rows is None:
            order = [int(n) for n in sorted_df["nums"]]
            labels = [f"{i + 1:02d}" for i in range(len(order))]
            placement = ["sequential"] * len(order)
        else:
            order, labels, placement, report = self._assignLayoutLabels(sorted_df, boxes.orig_shape)

        final = keep[np.array(order, dtype=int)]
        return FruitInstances(
            boxes=data[final],
            masks=mask_data[final],
            labels=labels,
            placement=placement,
            rejected=[(box, reason, mask_data[position]) for box, reason, position in rejected],
            report=report,
        )

    @staticmethod
    def _putOutlinedText(
        image: NDArray[np.uint8],
        text: str,
        origin: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Draws text with a dark outline so it can be read on any color.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, origin, font, scale, (0, 0, 0), thickness + 8)
        cv2.putText(image, text, origin, font, scale, color, thickness)

    def _extractMaskedImage(self, tray_image: Image, instances: Optional[FruitInstances] = None) -> Image:
        """
        Extracts a masked image from the given tray image, overlaying segmentation masks and bounding boxes.

        This method applies segmentation masks to the input tray image and draws bounding boxes around the detected fruit.
        Each mask is colored uniquely. The fruit number is drawn large in the middle of each fruit and matches the
        NN in the segmented file name ``..._fruit_NN.png``. The confidence score is drawn at the top-left of the box.
        Detections discarded as false detections are not drawn (they are on the unfiltered masked image, see
        `_extractFullMaskedImage`, and saved as images, see `_extractFalsePositiveImages`). In research mode, a
        tray image that needs attention also gets a line of text along the bottom explaining why.

        Args:
            tray_image (Image): The input tray image from which to extract the masked image.
                This object is expected to have methods `getSegmentationResults()` and `getImage()`.
            instances (FruitInstances): The fruit to draw. Found with `_getFruitInstances` when not given.

        Returns:
            Image: An image instance with the segmentation masks and bounding boxes overlaid.
        """
        if instances is None:
            instances = self._getFruitInstances(tray_image)
        if instances is None:
            raise AttributeError("No fruit found to draw on the masked image.")

        result = tray_image.getImage().copy()
        height, width = result.shape[:2]
        alpha = self.mask_alpha.getValue()
        num_instances = len(instances.labels)
        brightness = self.color_brightness.getValue()
        hsv = [(i / num_instances, 1, brightness) for i in range(num_instances)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.font_scale.getValue()
        text_thickness = self.text_thickness.getValue()
        for i in range(num_instances):
            mask = instances.masks[i]
            (r, g, b) = colors[i]
            for c in range(3):
                result[:, :, c] = np.where(
                    mask == 1,
                    result[:, :, c] * (1 - alpha) + alpha * colors[i][c] * 255,
                    result[:, :, c],
                )

            x1, y1, x2, y2, conf, _ = instances.boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(result, (x1, y1), (x2, y2), (r * 255, g * 255, b * 255), self.bbox_thickness.getValue())

            # confidence at the top-left of the box, kept inside the image
            conf_text = "{:.3f}".format(conf)
            (_, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, text_thickness)
            cv2.putText(
                result,
                conf_text,
                (max(x1, 0), max(y1, conf_h + 5)),
                fontFace=font,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=text_thickness,
            )

            # fruit number (same as the NN in "_fruit_NN.png") in the middle of the fruit; a fruit
            # whose position on the tray is not known is drawn in red
            label = instances.labels[i]
            num_scale = font_scale * 2.5
            num_thick = text_thickness * 2
            (num_w, num_h), _ = cv2.getTextSize(label, font, num_scale, num_thick)
            box_w = max(x2 - x1, 1)
            if num_w > 0.85 * box_w:  # long labels are shrunk to stay inside their fruit
                shrink = 0.85 * box_w / num_w
                num_scale *= shrink
                num_thick = max(1, int(round(num_thick * shrink)))
                (num_w, num_h), _ = cv2.getTextSize(label, font, num_scale, num_thick)
            origin = ((x1 + x2) // 2 - num_w // 2, (y1 + y2) // 2 + num_h // 2)
            color = (0, 0, 255) if instances.placement[i] == "unplaced" else (255, 255, 255)
            self._putOutlinedText(result, label, origin, num_scale, color, num_thick)

        # research mode: say why this tray image needs attention, shrunk to fit the image width
        banner = instances.report.get("banner", "")
        if banner:
            scale = font_scale
            (text_w, _), _ = cv2.getTextSize(banner, font, scale, text_thickness)
            if text_w > width - 40:
                scale = max(0.3, scale * (width - 40) / text_w)
            self._putOutlinedText(result, banner, (20, height - 20), scale, (0, 215, 255), text_thickness)

        image_instance: Image = RGBImage(
            pathlib.Path(tray_image.getImageName()).stem + f"_masked_image" + ".png"
        )
        image_instance.setImage(result)
        return image_instance

    def _extractFullMaskedImage(self, tray_image: Image) -> Optional[Image]:
        """
        Draws everything the model found on the tray image, before any filtering or numbering.

        This is the picture Granny has always written to full_masked_images: every detection (fruit
        or tray information, true or false) gets a colored mask, a bounding box and its confidence
        score. There is deliberately no fruit number here; the numbered picture with the false
        detections removed is the one in full_masked_images_clean. The image is turned the same way
        as the clean one, so the two can be compared directly.

        Args:
            tray_image (Image): The tray image with segmentation results already set.

        Returns:
            The masked image, or None when the model found nothing on this tray image.
        """
        [results] = tray_image.getSegmentationResults()
        if results.boxes is None or results.masks is None:
            return None
        data = results.boxes.cpu().numpy().data  # (N, 6): x1, y1, x2, y2, conf, cls
        mask_data = results.masks.cpu().numpy().data  # (N, H, W)
        num_instances = len(data)
        if num_instances == 0:
            return None

        result = tray_image.getImage().copy()
        alpha = self.mask_alpha.getValue()
        brightness = self.color_brightness.getValue()
        colors = [colorsys.hsv_to_rgb(i / num_instances, 1, brightness) for i in range(num_instances)]
        random.shuffle(colors)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.font_scale.getValue()
        text_thickness = self.text_thickness.getValue()
        for i in range(num_instances):
            mask = mask_data[i]
            (r, g, b) = colors[i]
            for c in range(3):
                result[:, :, c] = np.where(
                    mask == 1,
                    result[:, :, c] * (1 - alpha) + alpha * colors[i][c] * 255,
                    result[:, :, c],
                )
            x1, y1, x2, y2, conf, _ = data[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(result, (x1, y1), (x2, y2), (r * 255, g * 255, b * 255), self.bbox_thickness.getValue())
            conf_text = "{:.3f}".format(conf)
            (_, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, text_thickness)
            cv2.putText(
                result,
                conf_text,
                (max(x1, 0), max(y1, conf_h + 5)),
                fontFace=font,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=text_thickness,
            )

        image_instance: Image = RGBImage(pathlib.Path(tray_image.getImageName()).stem + "_masked_image" + ".png")
        image_instance.setImage(result)
        return image_instance

    def _extractFalsePositiveImages(self, tray_image: Image, instances: FruitInstances) -> List[Image]:
        """
        Cuts out each detection that was discarded as a false detection, so a person can always go
        back and check whether the filter was right.

        The images are made exactly like the fruit images (the detection's mask applied to its
        bounding box) and are named ``<image>_false_positive_NN.png``, numbered 01, 02, ... in
        order of position on the tray image (top to bottom, then left to right). What was discarded,
        and why, is also recorded in false_positives.csv.

        Args:
            tray_image (Image): The tray image the detections came from.
            instances (FruitInstances): The result of `_getFruitInstances` for that image.

        Returns:
            One image per discarded detection (an empty list when nothing was discarded).
        """
        tray_array: NDArray[np.uint8] = tray_image.getImage()
        height, width = tray_array.shape[:2]
        stem = pathlib.Path(tray_image.getImageName()).stem
        discarded = sorted(instances.rejected, key=lambda item: (item[0][1], item[0][0]))

        images: List[Image] = []
        for number, (box, reason, mask) in enumerate(discarded, start=1):
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
            file_name = f"{stem}_false_positive_{number:02d}.png"
            self._false_positive_rows.append(
                {
                    "image": tray_image.getImageName(),
                    "file": file_name,
                    "reason": reason,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": round(float(box[4]), 3),
                }
            )
            if x2 <= x1 or y2 <= y1:
                continue
            crop = np.zeros([y2 - y1, x2 - x1, 3], dtype=np.uint8)
            for channel in range(3):
                crop[:, :, channel] = tray_array[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_instance: Image = RGBImage(file_name)
            image_instance.setImage(crop)
            images.append(image_instance)
        return images

    def _sortInstances(self, boxes: NDArray[np.float32], img_shape: Tuple[int, int]):
        """
        Helper function to sort the fruit tray using their center coordinates.

        This sorting algorithm follows the numbering convention in demo/numbering_tray_convention.pdf.
        In an increasing order, it sorts by y-center coordinates and then by x-center coordinates.

        Args:
            boxes (NDArray[np.float32]): A NumPy array of shape (N, 6), where N is the number of bounding boxes.
                Each row represents a bounding box with the format [x1, y1, x2, y2, conf, cls].
            img_shape (Tuple[int, int]): A tuple representing the shape of the image (height, width).

        Returns:
            pd.DataFrame: A DataFrame containing the sorted bounding boxes with additional columns:
                - ycenter: The y-coordinate of the center of the bounding box.
                - xcenter: The x-coordinate of the center of the bounding box.
                - rows: The row number assigned based on the y-center coordinates.
                - apple_id: The unique identifier assigned to each bounding box after sorting.
        """
        h, _ = img_shape
        df = pd.DataFrame(boxes)
        df.columns = ["x1", "y1", "x2", "y2", "conf", "cls"]
        df["ycenter"] = ((df["y1"] + df["y2"]) / 2).astype(int)
        df["xcenter"] = ((df["x1"] + df["x2"]) / 2).astype(int)
        df["rows"] = 0
        df["apple_id"] = 0
        df["nums"] = df.index
        df = df.sort_values("ycenter", ascending=True).reset_index(drop=True)
        df["rows"] = (df["ycenter"].diff().abs().gt(h // self.row_tolerance.getValue()).cumsum() + 1).fillna(1).astype(int)

        df_list: List[pd.DataFrame] = []
        apple_id = 1
        increment = 1
        for i in range(1, df["rows"].max() + 1):
            dfx = (
                df[df["rows"] == i].sort_values("xcenter", ascending=False).reset_index(drop=True)
            )
            dfx["apple_id"] = range(apple_id, apple_id + increment * len(dfx), increment)
            df_list.append(dfx)
            apple_id += increment * len(dfx)

        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df

    def _extractTrayInfo(self, tray_image: Image) -> List[Image]:
        """
        Extracts individual tray information instances from the given tray image.

        This method identifies and isolates instances of the "tray_info" class within the provided tray image.
        It uses the segmentation results to extract bounding boxes and masks for the relevant class,
        sorts these instances, and then extracts and saves individual images for each detected instance.

        Args:
            tray_image (Image): The input tray image from which to extract tray information instances.
                This object is expected to have methods `getSegmentationResults()` and `getImage()`.

        Returns:
            List[Image]: A list of `Image` objects representing the individual tray information instances.
                If no instances of the "tray_info" class are found, an empty list is returned.
        """
        info_cls = SegmentationConfig.CLASSES["tray_info"]

        # gets bounding boxes, binary masks, and the original full-tray image array
        [results] = tray_image.getSegmentationResults()
        boxes = results.boxes.cpu().numpy()
        masks = results.masks.cpu().numpy()
        # checks for tray info class in segmentation results
        if not info_cls in boxes.cls:
            return []
        tray_idx: NDArray[int] = np.where(boxes.cls == info_cls)  # type:ignore
        boxes = boxes[tray_idx]
        masks = masks[tray_idx]
        tray_image_array: NDArray[np.uint8] = tray_image.getImage()

        # sorts boxes and masks based on xy-coordinates
        sorted_df = self._sortInstances(boxes.data, boxes.orig_shape)
        order: NDArray[np.float32] = sorted_df["nums"].to_numpy()
        sorted_boxes = boxes.data[order]  # type: ignore
        sorted_masks = masks.data[order]  # type: ignore

        # extracts instances based on bounding boxes and masks
        tray_images: List[Image] = []
        for i in range(len(sorted_masks)):
            x1, y1, x2, y2, _, _ = sorted_boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            individual_image = np.zeros([y2 - y1, x2 - x1, 3], dtype=np.uint8)
            mask = sorted_masks[i]
            for channel in range(3):
                individual_image[:, :, channel] = tray_image_array[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_name = (
                pathlib.Path(tray_image.getImageName()).stem + f"_tray_info_{i+1:02d}" + ".png"
            )
            image_instance: Image = RGBImage(image_name)
            image_instance.setImage(individual_image)
            tray_images.append(image_instance)

        # returns a list of individual instances
        return tray_images

    def _extractImage(self, tray_image: Image, instances: Optional[FruitInstances] = None) -> List[Image]:
        """
        Extracts individual fruit instances from the given tray image using binary masks.

        This method performs instance segmentation on the input tray image to isolate and extract
        images of each fruit detected by the YOLO model. It utilizes the segmentation results to
        obtain bounding boxes and masks, sorts the instances, and then extracts and saves individual
        images for each detected fruit. Each file is named after the fruit's number (or its position
        label in research mode), which is the same number drawn on the full masked image.

        Args:
            tray_image (Image): The input tray image from which to extract fruit instances.
                This object is expected to have methods `getSegmentationResults()` and `getImage()`.
            instances (FruitInstances): The fruit to extract. Found with `_getFruitInstances` when not given.

        Returns:
            List[Image]: A list of `Image` objects representing the individual fruit instances.
                If no instances of the "fruits" class are found, an empty list is returned.
        """
        # gets the filtered, numbered fruit detections and the original full-tray image array.
        # The same fruit and numbers are drawn on the full masked image.
        if instances is None:
            instances = self._getFruitInstances(tray_image)
        if instances is None:
            return []
        sorted_boxes, sorted_masks = instances.boxes, instances.masks
        tray_image_array: NDArray[np.uint8] = tray_image.getImage()

        # extracts instances based on bounding boxes and masks
        individual_images: List[Image] = []
        for i in range(len(sorted_masks)):
            x1, y1, x2, y2, _, _ = sorted_boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            individual_image = np.zeros([y2 - y1, x2 - x1, 3], dtype=np.uint8)
            mask = sorted_masks[i]
            for channel in range(3):
                individual_image[:, :, channel] = tray_image_array[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_name = pathlib.Path(tray_image.getImageName()).stem + f"_fruit_{instances.labels[i]}" + ".png"
            image_instance: Image = RGBImage(image_name)
            image_instance.setImage(individual_image)
            individual_images.append(image_instance)

        # returns a list of individual instances
        return individual_images

    def performAnalysis(self) -> List[Image]:
        """
        Runs the full segmentation pipeline on user-provided images.

        Steps:
        - Downloads or loads pretrained model.
        - Loads and rotates images if needed.
        - Runs YOLO segmentation.
        - Extracts:
            - Individual fruit images
            - Tray info (labels, QR)
            - Masked overlays
        - Saves all outputs to results folders.

        Returns:
            ist[Image]: All segmented fruit image instances.
        """
        # checks the options (and reads the tray layout in research mode) before doing any work
        self._checkOptions()
        self._manifest_rows = []
        self._false_positive_rows = []
        self._report_rows = []
        self._orientation_info = {}
        self._reference_samples = []

        self.model_name: str = self.in_params.get(self.model.getName()).getValue()  # type:ignore

        # download trained ML models from https://osf.io to the current directory
        if self.model_name.endswith(".pt"):
            self.local_model_path = self.model_name
        else:
            self.local_model_path = os.path.join(
                f"{pathlib.Path(__file__).parent}",
                self.models[self.model_name]["full_name"],
            )
            if not os.path.exists(self.local_model_path):
                model_url: str = self._getModelUrl(self.model_name)  # type: ignore
                self._downloadTrainedWeights(model_url)

        # loads segmentation model
        self.AIModel: AIModel = YoloModel(self.local_model_path)
        self.AIModel.loadModel()
        self.segmentation_model = self.AIModel.getModel()

        # initiates user's input
        self.input_images = self.in_params.get(self.input_images.getName())  # type:ignore

        # initiates Granny.Model.Images.Image instances for the analysis using the user's input
        self.input_images.readValue()
        self.images = self.input_images.getImageList()

        # initiates ImageIO
        self.image_io: ImageIO = RGBImageFile()

        # performs segmentation on each image one-by-one
        segmented_images: List[Image] = []
        for image_instance in self.images:
            # set ImageIO with specific file path
            self.image_io.setFilePath(image_instance.getFilePath())

            # loads image from file system with RGBImageFile(ImageIO)
            image_instance.loadImage(image_io=self.image_io)

            # turns the image as requested (by default, portrait images are rotated to landscape)
            turned = self._orientImage(image_instance)
            orientation_note, certain = "", True

            # predicts fruit instances in the image
            result = self._segmentInstances(image=image_instance.getImage())

            # sets segmentation result
            image_instance.setSegmentationResults(results=result)

            # research mode with orientation 'auto': pick the turn that fits the tray layout, and
            # find the fruit again only if the image has to be turned
            if self.layout_rows is not None and self.orientation.getValue() == "auto":
                extra, orientation_note, certain = self._chooseRotation(image_instance)
                if extra:
                    code = {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    }[extra]
                    image_instance.setImage(cv2.rotate(image_instance.getImage(), code))
                    result = self._segmentInstances(image=image_instance.getImage())
                    image_instance.setSegmentationResults(results=result)
                    turned = (turned + extra) % 360
            self._orientation_info[image_instance.getImageName()] = (turned, orientation_note, certain)

            try:
                # the unfiltered picture of everything the model found goes to "full_masked_images"
                # first, so it is there even when no fruit is found
                full_masked_image = self._extractFullMaskedImage(image_instance)
                if full_masked_image is not None:
                    self.masked_images.setImageList([full_masked_image])
                    self.masked_images.writeValue()

                # finds, filters and numbers the fruit once, for the fruit images and the clean masked image
                instances = self._getFruitInstances(image_instance)
                if instances is None:
                    print(f"{image_instance.getImageName()}: no fruit found")
                    self._recordResults(image_instance, None)
                    continue

                # extracts individual instances
                image_instances = self._extractImage(image_instance, instances)
                # and tray information
                tray_infos = self._extractTrayInfo(image_instance)
                # and the clean masked image (numbered, false detections removed)
                clean_masked_image = self._extractMaskedImage(image_instance, instances)
                # and the detections that were discarded as false detections
                false_positive_images = self._extractFalsePositiveImages(image_instance, instances)

                # 1. sets the output ImageListValue with the list of segmented images
                # 2. writes the segmented images to "segmented_images" folder
                # 3. writes the tray information to "tray_infos" folder
                # 4. writes the clean masked image to "full_masked_images_clean" folder
                # 5. writes the discarded detections to "segmented_images_false_positive" folder
                self.seg_images.setImageList(image_instances)
                self.seg_images.writeValue()

                self.tray_infos.setImageList(tray_infos)
                self.tray_infos.writeValue()

                self.masked_images_clean.setImageList([clean_masked_image])
                self.masked_images_clean.writeValue()

                if false_positive_images:
                    self.seg_images_false_positive.setImageList(false_positive_images)
                    self.seg_images_false_positive.writeValue()

                self._recordResults(image_instance, instances)
            except Exception as err:
                print(f"{image_instance.getImageName()}: results could not be written ({type(err).__name__}: {err})")

        self._writeReports()
        return segmented_images

    def _recordResults(self, tray_image: Image, instances: Optional[FruitInstances]) -> None:
        """
        Remembers what was found on one tray image, for the csv files written at the end.

        In research mode, a tray image that needs attention is also announced in the console.
        """
        name = tray_image.getImageName()
        stem = pathlib.Path(name).stem
        if instances is not None:
            for i, label in enumerate(instances.labels):
                x1, y1, x2, y2, conf, _ = instances.boxes[i]
                self._manifest_rows.append(
                    {
                        "image": name,
                        "file": f"{stem}_fruit_{label}.png",
                        "label": label,
                        "placement": instances.placement[i],
                        "x_center": int((x1 + x2) / 2),
                        "y_center": int((y1 + y2) / 2),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "confidence": round(float(conf), 3),
                    }
                )

        if self.layout_rows is None:
            return
        report = instances.report if instances is not None else {}
        if instances is None:
            report = {
                "detected": 0,
                "expected": int(self.nexpect.getValue()),
                "status": "check",
                "missing": [],
                "unplaced": 0,
                "notes": "no fruit found",
            }
        turned, orientation_note, _ = self._orientation_info.get(name, (0, "", True))
        notes = "; ".join(n for n in (report["notes"], orientation_note) if n)
        self._report_rows.append(
            {
                "image": name,
                "detected": report["detected"],
                "expected": report["expected"],
                "status": report["status"],
                "empty_positions": " ".join(report["missing"]),
                "unplaced_fruit": report["unplaced"],
                "turned_degrees_clockwise": turned,
                "notes": notes,
            }
        )
        if report["status"] != "ok":
            print(f"NOTICE {name}: {report['notes']}")

        # a tray whose fruit were all placed without doubt (and the right way up) can teach the reference
        if (
            instances is not None
            and report["status"] in ("ok", "fewer")
            and report["unplaced"] == 0
            and "reference" not in instances.placement
            and self._orientation_info.get(name, (0, "", True))[2]
        ):
            image_height, image_width = tray_image.getShape()[:2]
            self._reference_samples.append(
                {
                    label: ((box[0] + box[2]) / 2 / image_width, (box[1] + box[3]) / 2 / image_height)
                    for label, box in zip(instances.labels, instances.boxes)
                }
            )

    def _writeReports(self) -> None:
        """
        Writes fruit_manifest.csv (every fruit file with its label and position on the image),
        false_positives.csv (every discarded detection with the reason) and, in
        research mode, numbering_report.csv (one line per tray image with the count check), next to
        the segmented_images folder.
        """
        out_dir = os.path.dirname(self.seg_images.getValue())
        if self._manifest_rows:
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(self._manifest_rows).to_csv(os.path.join(out_dir, "fruit_manifest.csv"), index=False)
        if self._false_positive_rows:
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(self._false_positive_rows).to_csv(os.path.join(out_dir, "false_positives.csv"), index=False)
        if self.layout_rows is not None and self._report_rows:
            os.makedirs(out_dir, exist_ok=True)
            pd.DataFrame(self._report_rows).to_csv(os.path.join(out_dir, "numbering_report.csv"), index=False)
            flagged = [row["image"] for row in self._report_rows if row["status"] != "ok"]
            combined = buildTrayReference(self._reference_samples)
            if combined:
                reference_path = os.path.join(out_dir, "tray_reference.csv")
                pd.DataFrame(
                    [
                        {"label": label, "x": round(combined[label][0], 5), "y": round(combined[label][1], 5), "n_images": combined[label][2]}
                        for row in (self.layout_rows or [])
                        for label in row
                        if label in combined
                    ]
                ).to_csv(reference_path, index=False)
                print(
                    f"\nTray reference saved to {reference_path} ({len(combined)} of "
                    f"{int(self.nexpect.getValue())} positions, from {len(self._reference_samples)} tray images). "
                    "It is best made from photos of full trays; use it with --tray_reference."
                )
            if self.orientation.getValue() == "auto":
                turned = [n for n, (deg, _, _) in self._orientation_info.items() if deg]
                unsure = [n for n, (_, _, certain) in self._orientation_info.items() if not certain]
                print(
                    f"\nOrientation: {len(turned)} of {len(self._orientation_info)} images were turned to fit "
                    "the tray layout (column turned_degrees_clockwise in numbering_report.csv)."
                )
                if unsure:
                    print(f"  {len(unsure)} image(s) could not be oriented with certainty (see notes in the report).")
            print(
                f"\nResearch check: {len(self._report_rows) - len(flagged)} of {len(self._report_rows)} tray "
                f"images have all {int(self.nexpect.getValue())} fruit placed."
            )
            if flagged:
                print("Needs attention (see numbering_report.csv and the red/yellow text on the masked images):")
                for name in flagged:
                    print(f"  {name}")

    def _preRun(self):
        """
        Stub implementation - Segmentation overrides performAnalysis() entirely.
        """
        pass

    def _processImage(self, image_instance: Image) -> Image:
        """
        Stub implementation - Segmentation overrides performAnalysis() entirely.
        """
        pass

    def _postRun(self, results):
        """
        Stub implementation - Segmentation overrides performAnalysis() entirely.
        """
        pass
