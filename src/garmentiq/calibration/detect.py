# garmentiq/calibration/detect.py
"""Detecting ArUco markers with sub-pixel corner accuracy.

Every millimetre of accuracy downstream rests on where these corners land, so the
detector always runs with sub-pixel refinement: without it corners snap to whole
pixels, which at a metre of garment is already worth more than the tolerance.

The board is hand-made, so which dictionary it uses may not be written down.
`identify_dictionary` finds out by trying all of them on a photo.
"""
from pathlib import Path

import cv2
import numpy as np

if not hasattr(cv2, "aruco"):  # pragma: no cover - environment guard
    raise ImportError(
        f"This OpenCV build ({cv2.__version__}) has no cv2.aruco module. ArUco lives in "
        f"the main package from OpenCV 5 onward; on OpenCV 4 it is contrib-only. "
        f"Install opencv-python>=5.0.0 (or opencv-contrib-python, which cannot be "
        f"installed alongside opencv-python)."
    )

# Every predefined dictionary this OpenCV build knows, newest names first. Used by
# identify_dictionary; duplicates such as DICT_APRILTAG_36h11 / _36H11 are collapsed.
PREDEFINED_DICTIONARIES = {
    name: getattr(cv2.aruco, name)
    for name in sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
}
DICTIONARY_NAMES = {value: name for name, value in sorted(PREDEFINED_DICTIONARIES.items())}


class DetectionError(ValueError):
    """Raised when marker detection cannot be trusted."""


def dictionary_name(dictionary):
    """Returns the readable name of a predefined dictionary id, e.g. 'DICT_4X4_50'."""
    return DICTIONARY_NAMES.get(int(dictionary), f"dictionary {int(dictionary)}")


def load_image(image_path):
    """Reads a BGR image from disk.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def as_gray(image):
    """Returns a single-channel view of a BGR or already-grayscale image."""
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_detector(dictionary, corner_refine_win_size=5):
    """Creates an ArUco detector with sub-pixel corner refinement enabled.

    Args:
        dictionary (int): OpenCV predefined ArUco dictionary id.
        corner_refine_win_size (int): Half-width of the refinement window, in pixels.
            Lower it if a marker is under roughly 60 px wide in the photos, or the
            window will reach past the marker and drag corners off.

    Returns:
        cv2.aruco.ArucoDetector: Detector ready for `detect_markers`.
    """
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = int(corner_refine_win_size)
    parameters.cornerRefinementMaxIterations = 100
    parameters.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(int(dictionary)), parameters
    )


def detect_markers(image, dictionary=None, keep_ids=None, detector=None):
    """Detects markers and returns their sub-pixel corners.

    Args:
        image (str | pathlib.Path | numpy.ndarray): Image path, or a BGR/gray image.
        dictionary (int, optional): Predefined dictionary id. Ignored when `detector`
            is given; one of the two is required.
        keep_ids (iterable, optional): Keep only these marker ids, ignore any others.
            Use it to shut out stray markers that are not part of the board.
        detector (cv2.aruco.ArucoDetector, optional): Reuse a detector across images,
            which is much faster than building one per frame.

    Returns:
        dict: Marker id -> (4, 2) float64 corners, in OpenCV order (top-left,
        top-right, bottom-right, bottom-left) as seen in the image.

    Raises:
        ValueError: If neither `dictionary` nor `detector` is given.
        DetectionError: If the same marker id is found more than once, which means
            two boards, or a reflection, are in view.
    """
    if detector is None:
        if dictionary is None:
            raise ValueError("Pass either a dictionary id or a ready-made detector.")
        detector = create_detector(dictionary)
    if isinstance(image, (str, Path)):
        image = load_image(image)

    corners, ids, _ = detector.detectMarkers(as_gray(image))
    found = {}
    if ids is None:
        return found

    keep_ids = None if keep_ids is None else {int(i) for i in keep_ids}
    for marker_id, marker_corners in zip(ids.ravel(), corners):
        marker_id = int(marker_id)
        if keep_ids is not None and marker_id not in keep_ids:
            continue
        if marker_id in found:
            raise DetectionError(
                f"Marker id {marker_id} was detected twice in the same image. Keep "
                f"only one board in view, and check for a reflective surface."
            )
        found[marker_id] = marker_corners.reshape(4, 2).astype(np.float64)
    return found


def identify_dictionary(image, candidates=None, min_markers=2):
    """Works out which ArUco dictionary a board uses, by trying them all.

    Args:
        image (str | pathlib.Path | numpy.ndarray): A photo of the bare board.
        candidates (iterable, optional): Dictionary ids to try; defaults to every
            predefined dictionary in this OpenCV build.
        min_markers (int): Fewest markers a dictionary must find to be reported.

    Returns:
        list: `{"dictionary", "name", "marker_ids", "n_markers"}` for each dictionary
        that found at least `min_markers`, best first. Normally one entry; several
        means the dictionaries overlap and you should pick the one whose ids look
        like your board.
    """
    if isinstance(image, (str, Path)):
        image = load_image(image)
    gray = as_gray(image)

    if candidates is None:
        candidates = sorted(set(PREDEFINED_DICTIONARIES.values()))

    results = []
    for dictionary in candidates:
        try:
            found = detect_markers(gray, dictionary=dictionary)
        except DetectionError:
            continue  # duplicate ids: this dictionary is misreading the board
        if len(found) >= min_markers:
            results.append(
                {
                    "dictionary": int(dictionary),
                    "name": dictionary_name(dictionary),
                    "marker_ids": sorted(found),
                    "n_markers": len(found),
                }
            )
    results.sort(key=lambda result: -result["n_markers"])
    return results


def detect_in_images(image_paths, dictionary, keep_ids=None, corner_refine_win_size=5):
    """Detects markers in many images with one shared detector.

    Args:
        image_paths (iterable): Image paths.
        dictionary (int): Predefined dictionary id.
        keep_ids (iterable, optional): Keep only these marker ids.
        corner_refine_win_size (int): Passed to `create_detector`.

    Returns:
        list: `{"name", "corners", "image_size"}` per image, skipping images where no
        marker was found. `corners` is the dict from `detect_markers`, `image_size` is
        (width, height).
    """
    detector = create_detector(dictionary, corner_refine_win_size)
    views = []
    for image_path in image_paths:
        image = load_image(image_path)
        corners = detect_markers(image, keep_ids=keep_ids, detector=detector)
        if not corners:
            continue
        height, width = image.shape[:2]
        views.append(
            {
                "name": Path(image_path).name,
                "corners": corners,
                "image_size": (width, height),
            }
        )
    return views
