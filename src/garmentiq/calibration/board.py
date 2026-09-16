# garmentiq/calibration/board.py
"""Holding the millimetre layout of a calibration board.

A `BoardSpec` says where every marker corner sits on the board, in millimetres. The
board here is hand-made, so there is no grid formula: the layout is a plain
`{marker id: 4 corners}` table, solved from photos by `learn_board` and saved to JSON.
Markers may sit at irregular positions and at any rotation.

Board coordinates are millimetres with X to the right and Y down, matching image
coordinates, so a homography between them has no flip. The origin is wherever
`learn_board` put it, which is arbitrary: distances never depend on it.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The four corners of a marker, in the order OpenCV's ArUco detector reports them.
CORNER_ORDER = ("top-left", "top-right", "bottom-right", "bottom-left")


@dataclass
class BoardSpec:
    """Where every marker of a calibration board sits, in millimetres.

    Attributes:
        dictionary (int): OpenCV predefined ArUco dictionary id the board uses.
        marker_size_mm (float): Nominal side of one printed marker square, including
            its black border. The single absolute scale reference of the system.
        corners_mm (dict): Marker id -> (4, 2) float64 corner positions in board
            millimetres, in OpenCV order (top-left, top-right, bottom-right,
            bottom-left) as seen on the upright board.
        diagnostics (dict): How the layout was obtained and how well it fits; written
            by `learn_board`, empty for a hand-built spec.
    """

    dictionary: int
    marker_size_mm: float
    corners_mm: dict
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.corners_mm = {
            int(marker_id): np.asarray(corners, dtype=np.float64).reshape(4, 2)
            for marker_id, corners in self.corners_mm.items()
        }
        if not self.corners_mm:
            raise ValueError("A board needs at least one marker.")
        self.marker_size_mm = float(self.marker_size_mm)
        self.dictionary = int(self.dictionary)

    @property
    def marker_ids(self):
        """Sorted list of the marker ids on the board."""
        return sorted(self.corners_mm)

    @property
    def extent_mm(self):
        """(width, height) of the bounding box of every marker corner."""
        points = np.vstack(list(self.corners_mm.values()))
        return tuple((points.max(axis=0) - points.min(axis=0)).tolist())

    def points_mm(self, marker_ids):
        """Stacks the (4, 2) corners of `marker_ids` into one (4N, 2) array, in order.

        Args:
            marker_ids (iterable): Marker ids to stack.

        Returns:
            numpy.ndarray: (4N, 2) float64 board coordinates in millimetres.

        Raises:
            KeyError: If an id is not on the board.
        """
        return np.vstack([self.corners_mm[int(i)] for i in marker_ids])

    def object_points(self, marker_ids):
        """Same as `points_mm`, with a zero Z column, for OpenCV's 3D calibration calls.

        Returns:
            numpy.ndarray: (4N, 3) float64 points on the Z = 0 board plane.
        """
        flat = self.points_mm(marker_ids)
        return np.column_stack([flat, np.zeros(len(flat))])

    def measured_marker_sizes_mm(self):
        """Side length implied by each marker's own corners: id -> mean of its 4 sides.

        A spread across markers means the board is not flat, or the layout was solved
        from too few views. Only meaningful for a learned board.
        """
        sizes = {}
        for marker_id, corners in self.corners_mm.items():
            sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
            sizes[marker_id] = float(sides.mean())
        return sizes

    def to_dict(self):
        return {
            "dictionary": int(self.dictionary),
            "marker_size_mm": float(self.marker_size_mm),
            "corners_mm": {
                str(marker_id): corners.tolist()
                for marker_id, corners in sorted(self.corners_mm.items())
            },
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            dictionary=data["dictionary"],
            marker_size_mm=data["marker_size_mm"],
            corners_mm=data["corners_mm"],
            diagnostics=data.get("diagnostics", {}),
        )

    def save_json(self, path):
        """Writes the board to JSON, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path):
        """Reads a board written by `save_json`.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No board file at {path}. Run the board learning step first "
                f"(python -m garmentiq.calibration.bootstrap)."
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def __str__(self):
        width, height = self.extent_mm
        return (
            f"BoardSpec({len(self.corners_mm)} markers, ids {self.marker_ids}, "
            f"{width:.1f} x {height:.1f} mm, marker {self.marker_size_mm:.2f} mm)"
        )
