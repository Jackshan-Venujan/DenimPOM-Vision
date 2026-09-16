# garmentiq/calibration/inspect_board.py
"""Reading a hand-made board off a single photo, before committing to a capture set.

Run this first. It answers the questions you would otherwise have to guess at: which
ArUco dictionary the board uses, which marker ids are on it, how they are arranged,
and whether they are being detected cleanly at all. Getting that wrong wastes a whole
capture session, and it costs one photo to check.

It also estimates the marker spacing, but only in units of the marker size — a photo
can show that one marker is 3.4 marker-widths from another, never that it is 170 mm,
because scale and distance are indistinguishable in a picture. That is why one marker
still has to be measured with a caliper.

    python -m garmentiq.calibration.inspect_board photo.jpg
"""
import argparse
from pathlib import Path

import numpy as np

from .detect import detect_markers, identify_dictionary, load_image
from .plot import draw_markers


def _marker_centres(corners_px):
    return {
        marker_id: np.asarray(corners, dtype=np.float64).mean(axis=0)
        for marker_id, corners in corners_px.items()
    }


def _mean_marker_side_px(corners):
    corners = np.asarray(corners, dtype=np.float64)
    return float(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1).mean())


def arrange_in_rows(corners_px, row_tolerance=0.6):
    """Groups markers into visual rows, top to bottom and left to right.

    Rows are found by clustering the marker centres vertically, with a tolerance in
    units of the marker size, so it works on a hand-made board whose rows are not
    perfectly aligned. Perspective tilts rows in a photo, so treat this as a readable
    summary, not as geometry — nothing downstream depends on it.

    Returns:
        list: Rows of marker ids, each row left to right.
    """
    if not corners_px:
        return []
    centres = _marker_centres(corners_px)
    scale = float(np.mean([_mean_marker_side_px(c) for c in corners_px.values()]))
    ordered = sorted(centres.items(), key=lambda item: item[1][1])

    rows, current, row_top = [], [], ordered[0][1][1]
    for marker_id, centre in ordered:
        if abs(centre[1] - row_top) > row_tolerance * scale:
            rows.append(current)
            current, row_top = [], centre[1]
        current.append((marker_id, centre))
    rows.append(current)
    return [[marker_id for marker_id, _ in sorted(row, key=lambda i: i[1][0])] for row in rows]


def estimate_spacing(corners_px):
    """Estimates neighbour spacing in units of the marker size.

    Perspective makes raw pixel distances unreliable, so each gap is divided by the
    mean marker size of the two markers involved, which cancels most of it.

    Returns:
        dict: `horizontal_marker_widths` and `vertical_marker_widths`, each the median
        centre-to-centre gap between nearest neighbours, or None if there is no pair.
    """
    if len(corners_px) < 2:
        return {"horizontal_marker_widths": None, "vertical_marker_widths": None}

    centres = _marker_centres(corners_px)
    sides = {i: _mean_marker_side_px(c) for i, c in corners_px.items()}
    horizontal, vertical = [], []

    ids = sorted(centres)
    for index, a in enumerate(ids):
        for b in ids[index + 1 :]:
            delta = centres[b] - centres[a]
            scale = 0.5 * (sides[a] + sides[b])
            dx, dy = abs(delta[0]) / scale, abs(delta[1]) / scale
            # Count a pair as a horizontal neighbour only when it is clearly sideways.
            if dx > dy * 2.0:
                horizontal.append(dx)
            elif dy > dx * 2.0:
                vertical.append(dy)

    def nearest(values):
        if not values:
            return None
        values = np.sort(np.asarray(values))
        # Neighbours are the smallest gaps; the rest are markers further along the row.
        near = values[values <= values[0] * 1.5]
        return round(float(np.median(near)), 3)

    return {
        "horizontal_marker_widths": nearest(horizontal),
        "vertical_marker_widths": nearest(vertical),
    }


def inspect(image_path, dictionary=None, overlay_path=None):
    """Reports everything a single photo can say about the board.

    Args:
        image_path (str | pathlib.Path): Photo of the bare board.
        dictionary (int, optional): Skip the dictionary sweep and use this one.
        overlay_path (str | pathlib.Path, optional): Where to write the annotated
            image. Defaults to `<photo>_detected.png` beside the photo.

    Returns:
        dict: `dictionary`, `name`, `marker_ids`, `rows`, `spacing`, `candidates`,
        `overlay_path` and `image_size`.

    Raises:
        ValueError: If no dictionary matches the photo.
    """
    image_path = Path(image_path)
    image = load_image(image_path)
    height, width = image.shape[:2]

    candidates = identify_dictionary(image) if dictionary is None else []
    if dictionary is None:
        if not candidates:
            raise ValueError(
                f"No ArUco dictionary matched {image_path.name}. Check the photo is "
                f"sharp and the markers are unobstructed and well lit."
            )
        dictionary = candidates[0]["dictionary"]

    from .detect import dictionary_name  # local import keeps the module's imports flat

    corners = detect_markers(image, dictionary=dictionary)
    overlay_path = Path(overlay_path) if overlay_path else image_path.with_name(
        f"{image_path.stem}_detected.png"
    )
    import cv2

    cv2.imwrite(str(overlay_path), draw_markers(image, corners))

    return {
        "dictionary": int(dictionary),
        "name": dictionary_name(dictionary),
        "marker_ids": sorted(corners),
        "rows": arrange_in_rows(corners),
        "spacing": estimate_spacing(corners),
        "candidates": candidates,
        "overlay_path": overlay_path,
        "image_size": (width, height),
    }


def main(argv=None):
    """Command line entry point: `python -m garmentiq.calibration.inspect_board`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="photo of the bare board")
    parser.add_argument("--dictionary", type=int, default=None,
                        help="skip the sweep and use this predefined dictionary id")
    parser.add_argument("--overlay", type=Path, default=None,
                        help="where to write the annotated image")
    args = parser.parse_args(argv)

    report = inspect(args.image, args.dictionary, args.overlay)
    width, height = report["image_size"]

    print(f"Image          : {args.image.name}  ({width} x {height})")
    print(f"Dictionary     : {report['name']}  (id {report['dictionary']})")
    if len(report["candidates"]) > 1:
        others = ", ".join(
            f"{c['name']} ({c['n_markers']})" for c in report["candidates"][1:]
        )
        print(f"  also matched : {others}")
    print(f"Markers found  : {len(report['marker_ids'])}  {report['marker_ids']}")

    print("Arrangement    :")
    for index, row in enumerate(report["rows"], start=1):
        print(f"  row {index}: {row}")

    spacing = report["spacing"]
    print("Spacing (in marker widths, relative only — caliper one marker for mm):")
    print(f"  horizontal   : {spacing['horizontal_marker_widths']}")
    print(f"  vertical     : {spacing['vertical_marker_widths']}")
    print(f"Overlay written: {report['overlay_path']}")

    print(
        "\nNext: set ARUCO_DICTIONARY in garmentiq/calibration/config.py to "
        f"cv2.aruco.{report['name']}, caliper one marker into MARKER_SIZE_MM, then "
        "capture a set with:\n"
        "  python -m garmentiq.calibration.capture --mode board"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
