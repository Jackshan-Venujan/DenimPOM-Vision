# garmentiq/calibration/learn_board.py
"""Solving where a hand-made board's markers sit, from photos of the board.

A hand-made board has no grid to type in, so its layout is measured instead. Because
the board is flat, each photo of it is related to the board by an exact homography,
and that is what makes the layout recoverable:

1. One marker is declared the origin. Its four corners *define* board millimetres,
   using the one number a photo cannot supply — the calipered marker size.
2. Every other marker is then placed by mapping its image corners through that
   photo's homography, growing outward across photos until all markers have a place.
3. The layout is refined by alternating between fitting each photo's homography to the
   current layout and re-averaging each marker's position over all photos. Many
   viewpoints beat the corner noise of any single one, which is what stops the error
   from growing toward the far edge of a large board.
4. After each pass every marker is snapped back to a perfect square of the known side.
   This re-injects the known scale at every marker rather than only at the origin, so
   scale cannot drift across the board.
5. Finally every marker pose and every photo's homography are optimised together.

That last step is what actually makes the layout accurate, and it is worth saying why
the first four are not enough on their own. Reprojection error alone cannot pin a
layout down: distort the whole board by any perspective transform, apply the inverse
to every photo's homography, and not a single pixel moves. Alternating between fitting
photos and re-averaging markers is blind to that drift, and only the square-snapping
pushes back — weakly, because a gentle perspective distortion across half a metre
still leaves every 50 mm marker looking square. The drift survives, and 0.01 px of
corner noise comes out as 0.2 mm of error.

Holding each marker as an *exact* square of known size removes the freedom entirely: no
perspective transform other than a rigid motion maps a set of equal squares to another
set of equal squares. So the joint fit parameterises each marker by three numbers
(turn, and where it sits) instead of eight free coordinates, fixes the reference marker
to nail the origin down, and minimises pixel reprojection over all of it at once.

What comes out is a `BoardSpec` plus the diagnostics to judge it: per-marker scatter
in millimetres, per-photo reprojection error in pixels, how many photos saw each
marker, and the size each marker turned out to be.
"""
import math

import cv2
import numpy as np

from .board import BoardSpec
from .undistort import undistort_marker_corners

# Corners of an upright marker of side s, in OpenCV order, centred on the origin.
_TEMPLATE_ORDER = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

# How much better the reprojection must get before a board is called mixed-size.
# Freeing the sizes always fits a little better, the way any extra parameter does;
# only a real difference in size buys this much.
_MIXED_SIZE_GAIN = 0.20

# Evaluation budget for the joint fit. It has to be generous: an unconverged fit and an
# overfitted one look alike from the outside, and a half-finished uniform fit will lose
# the size comparison below to a free-size fit that is merely further along.
_REFINE_BUDGET = 3000


class BoardLearningError(ValueError):
    """Raised when a board layout cannot be solved from the given photos."""


def _canonical_square(size_mm):
    """The (4, 2) corners of an upright marker with its top-left corner at the origin."""
    return (_TEMPLATE_ORDER + 0.5) * float(size_mm)


def _fit_square(corners_mm, size_mm):
    """Snaps four estimated corners onto a perfect upright-rotated square.

    Finds the rotation and translation (no scaling, no reflection) that best places a
    square of side `size_mm` on the four points, by the Kabsch algorithm.

    Returns:
        numpy.ndarray: (4, 2) float64 corners of the fitted square.
    """
    template = _TEMPLATE_ORDER * float(size_mm)
    centroid = corners_mm.mean(axis=0)
    centred = corners_mm - centroid

    # Kabsch, mapping the template onto the estimated corners. The covariance is
    # target-first; the other way round yields the inverse rotation.
    U, _, Vt = np.linalg.svd(centred.T @ template)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        # A reflection would mirror the marker; forbid it.
        U[:, -1] *= -1
        rotation = U @ Vt
    return template @ rotation.T + centroid


def _homography(source, destination, ransac_threshold_px):
    """Fits source -> destination, robustly when there are more than four points."""
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if len(source) < 4:
        return None
    method = 0 if len(source) == 4 else cv2.RANSAC
    H, _ = cv2.findHomography(source, destination, method, ransac_threshold_px)
    if H is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def _transform(points, H):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, H).reshape(-1, 2)


def _prepare_views(views, intrinsics):
    """Undistorts every view's corners and drops views with nothing detected."""
    prepared = []
    for index, view in enumerate(views):
        corners = view["corners"] if isinstance(view, dict) else view
        if not corners:
            continue
        name = view.get("name", f"view_{index}") if isinstance(view, dict) else f"view_{index}"
        prepared.append(
            {"name": name, "corners": undistort_marker_corners(corners, intrinsics)}
        )
    if not prepared:
        raise BoardLearningError("None of the given photos contain any markers.")
    return prepared


def _choose_reference(views, reference_id):
    """Picks the marker seen in the most photos, which anchors the whole layout."""
    counts = {}
    for view in views:
        for marker_id in view["corners"]:
            counts[marker_id] = counts.get(marker_id, 0) + 1

    if reference_id is not None:
        reference_id = int(reference_id)
        if reference_id not in counts:
            raise BoardLearningError(
                f"Reference marker {reference_id} does not appear in any photo; "
                f"markers seen were {sorted(counts)}."
            )
        return reference_id, counts
    # Most-seen marker wins; the lowest id breaks ties so the result is repeatable.
    return min(counts, key=lambda i: (-counts[i], i)), counts


def _seed_layout(views, reference_id, marker_size_mm, ransac_threshold_px):
    """Places markers in board millimetres, growing outward from the reference marker.

    Returns:
        dict: Marker id -> (4, 2) corners in millimetres.

    Raises:
        BoardLearningError: If some markers are never seen together with a placed one.
    """
    layout = {reference_id: _canonical_square(marker_size_mm)}

    # Each pass adds markers that share a photo with something already placed. A board
    # wider than one photo therefore needs overlapping photos to chain across.
    for _ in range(len(views) + 1):
        added = 0
        for view in views:
            known = [i for i in view["corners"] if i in layout]
            unknown = [i for i in view["corners"] if i not in layout]
            if not known or not unknown:
                continue
            image_points = np.vstack([view["corners"][i] for i in known])
            board_points = np.vstack([layout[i] for i in known])
            H = _homography(image_points, board_points, ransac_threshold_px)
            if H is None:
                continue
            for marker_id in unknown:
                layout[marker_id] = _fit_square(
                    _transform(view["corners"][marker_id], H), marker_size_mm
                )
                added += 1
        if added == 0:
            break

    seen = {marker_id for view in views for marker_id in view["corners"]}
    orphans = sorted(seen - set(layout))
    if orphans:
        raise BoardLearningError(
            f"Markers {orphans} were never photographed together with a marker that "
            f"could be placed. Add photos where they share the frame with the rest of "
            f"the board."
        )
    return layout


def _view_homographies(views, layout, ransac_threshold_px):
    """Fits board -> image for each photo, using every placed marker it can see.

    Fitting in this direction is the statistically right way round: the noise lives in
    the detected pixel corners, so that is what should be minimised.
    """
    fitted = []
    for view in views:
        ids = [i for i in view["corners"] if i in layout]
        if not ids:
            continue
        board_points = np.vstack([layout[i] for i in ids])
        image_points = np.vstack([view["corners"][i] for i in ids])
        H = _homography(board_points, image_points, ransac_threshold_px)
        if H is None:
            continue
        projected = _transform(board_points, H)
        errors = np.linalg.norm(projected - image_points, axis=1)
        fitted.append(
            {
                "name": view["name"],
                "ids": ids,
                "corners": view["corners"],
                "board_to_image": H,
                "image_to_board": np.linalg.inv(H),
                "rms_px": float(np.sqrt(np.mean(errors**2))),
            }
        )
    if not fitted:
        raise BoardLearningError("No photo could be fitted to the current layout.")
    return fitted


def _reestimate(fitted_views, layout):
    """Re-places every marker as the weighted mean of its position in each photo.

    A photo where the board fills the frame locates a marker far better than a distant
    one, so each photo's contribution is weighted by 1 / (mm per pixel)^2 — the
    inverse variance of the millimetre error its pixel noise produces.

    Returns:
        tuple: the new layout, and marker id -> per-photo scatter in millimetres.
    """
    sums, weights, samples = {}, {}, {}
    for view in fitted_views:
        for marker_id in view["ids"]:
            corners_mm = _transform(view["corners"][marker_id], view["image_to_board"])
            # Area scale of the image -> board map at this marker, in mm^2 per px^2.
            area_px = abs(_polygon_area(view["corners"][marker_id]))
            area_mm = abs(_polygon_area(corners_mm))
            mm_per_px_squared = area_mm / area_px if area_px > 0 else 1.0
            weight = 1.0 / max(mm_per_px_squared, 1e-12)

            sums[marker_id] = sums.get(marker_id, 0.0) + corners_mm * weight
            weights[marker_id] = weights.get(marker_id, 0.0) + weight
            samples.setdefault(marker_id, []).append(corners_mm)

    layout = {i: sums[i] / weights[i] for i in sums}
    scatter = {
        marker_id: float(
            np.sqrt(np.mean(np.linalg.norm(np.array(views) - layout[marker_id], axis=2) ** 2))
        )
        if len(views) > 1
        else 0.0
        for marker_id, views in samples.items()
    }
    return layout, scatter


def _polygon_area(corners):
    """Signed shoelace area of a quadrilateral."""
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _marker_pose(corners_mm):
    """Reads a marker's (turn, x, y) out of its four corners.

    The top edge runs from corner 0 to corner 1, so its direction is the marker's turn.
    """
    centroid = corners_mm.mean(axis=0)
    top_edge = corners_mm[1] - corners_mm[0]
    return math.atan2(float(top_edge[1]), float(top_edge[0])), centroid


def _pose_corners(angle, centre, size_mm):
    """Builds the four corners of an exact square from a (turn, x, y) pose."""
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return _TEMPLATE_ORDER * float(size_mm) @ rotation.T + np.asarray(centre)


def _refine_jointly(
    views,
    layout,
    sizes,
    reference_id,
    free_sizes=False,
    loss="huber",
    f_scale_px=1.0,
    max_nfev=400,
):
    """Optimises every marker pose and every photo's homography together.

    Each marker is three numbers — how it is turned and where it sits — at a fixed
    size, or four when `free_sizes` lets it find its own. The reference marker is never
    optimised: it fixes both the origin and, through its calipered size, the scale of
    everything else. Photos contribute eight homography numbers each. Minimising
    reprojection over all of it at once is what removes the perspective drift that
    alternating cannot see.

    Args:
        views (list): Prepared detections.
        layout (dict): Starting layout, marker id -> (4, 2) millimetres.
        sizes (dict): Starting size of each marker.
        reference_id (int): Marker held fixed.
        free_sizes (bool): Solve each marker's size as well as its pose. Use it when
            the board may carry markers of different sizes.
        loss (str): Robust loss for `scipy.optimize.least_squares`.
        f_scale_px (float): Residual size, in pixels, where the robust loss softens.
        max_nfev (int): Evaluation budget. The fit is at a hundredth of a pixel long
            before this, and the cap stops an over-parameterised free-size probe from
            grinding for minutes chasing nothing.

    Returns:
        tuple: refined layout, per-photo RMS in pixels, refined sizes, and the final
        cost — the last of which is what decides between a uniform and a mixed board.
    """
    from scipy.optimize import least_squares

    marker_ids = [i for i in sorted(layout) if i != reference_id]
    poses = {i: _marker_pose(layout[i]) for i in layout}
    per_marker = 4 if free_sizes else 3

    # Only photos that see at least one non-reference marker constrain anything new,
    # but every photo still helps place the markers it does see.
    usable = [view for view in views if any(i in layout for i in view["corners"])]
    homographies = []
    for view in usable:
        ids = [i for i in view["corners"] if i in layout]
        H = _homography(
            np.vstack([layout[i] for i in ids]),
            np.vstack([view["corners"][i] for i in ids]),
            3.0,
        )
        if H is None:
            return layout, {}, sizes, float("inf")
        homographies.append(H)

    # Pixel coordinates run to the thousands while the bottom row of a homography is
    # around 1e-5, and a trust region cannot step sensibly across eight orders of
    # magnitude. Working in image coordinates centred and scaled to about one unit puts
    # every homography entry near the same size. Because the mapping is a uniform
    # scale, multiplying the residual back by that scale gives exact pixels again.
    all_image_points = np.vstack(
        [view["corners"][i] for view in usable for i in sorted(view["corners"])
         if i in layout]
    )
    image_centre = all_image_points.mean(axis=0)
    image_scale = float(np.abs(all_image_points - image_centre).mean()) or 1.0
    normalise = np.array(
        [
            [1.0 / image_scale, 0.0, -image_centre[0] / image_scale],
            [0.0, 1.0 / image_scale, -image_centre[1] / image_scale],
            [0.0, 0.0, 1.0],
        ]
    )
    homographies = [normalise @ H for H in homographies]

    start = []
    for marker_id in marker_ids:
        angle, centre = poses[marker_id]
        start += [angle, float(centre[0]), float(centre[1])]
        if free_sizes:
            start.append(float(sizes[marker_id]))
    for H in homographies:
        start += (H / H[2, 2]).ravel()[:8].tolist()

    n_marker_params = per_marker * len(marker_ids)
    view_ids = [[i for i in sorted(view["corners"]) if i in layout] for view in usable]
    observed = (
        np.vstack(
            [np.vstack([view["corners"][i] for i in ids])
             for view, ids in zip(usable, view_ids)]
        )
        - image_centre
    ) / image_scale

    # The optimiser calls the residual thousands of times, so everything that does not
    # depend on the parameters is worked out once: where each view's rows start, and
    # which marker corner feeds each row. Rebuilding dictionaries and stacking arrays
    # per call is what made this take minutes rather than seconds.
    all_ids = [reference_id] + marker_ids
    row_of_marker = {marker_id: index for index, marker_id in enumerate(all_ids)}
    gather = np.concatenate(
        [
            np.arange(4) + 4 * row_of_marker[marker_id]
            for ids in view_ids
            for marker_id in ids
        ]
    )
    view_slices, start_row = [], 0
    for ids in view_ids:
        view_slices.append(slice(start_row, start_row + 4 * len(ids)))
        start_row += 4 * len(ids)

    reference_corners = _pose_corners(*poses[reference_id], sizes[reference_id])
    fixed_sizes = np.array([sizes[i] for i in marker_ids], dtype=np.float64)
    template_x, template_y = _TEMPLATE_ORDER[:, 0], _TEMPLATE_ORDER[:, 1]

    def unpack(params):
        block = params[:n_marker_params].reshape(len(marker_ids), per_marker)
        angles, centres = block[:, 0], block[:, 1:3]
        solved = np.abs(block[:, 3]) if free_sizes else fixed_sizes

        cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
        scale = solved[:, None]
        corners = np.empty((len(all_ids), 4, 2))
        corners[0] = reference_corners
        corners[1:, :, 0] = scale * (template_x * cos - template_y * sin) + centres[:, :1]
        corners[1:, :, 1] = scale * (template_x * sin + template_y * cos) + centres[:, 1:2]

        matrices = np.concatenate(
            [params[n_marker_params:].reshape(len(usable), 8), np.ones((len(usable), 1))],
            axis=1,
        ).reshape(len(usable), 3, 3)
        return corners, matrices, solved

    def residuals(params):
        corners, matrices, _ = unpack(params)
        points = corners.reshape(-1, 2)[gather]
        homogeneous = np.column_stack([points, np.ones(len(points))])

        projected = np.empty_like(points)
        for rows, H in zip(view_slices, matrices):
            mapped = homogeneous[rows] @ H.T
            projected[rows] = mapped[:, :2] / mapped[:, 2:3]
        # Back to pixels, so the robust loss and the reported RMS mean what they say.
        return ((projected - observed) * image_scale).ravel()

    result = least_squares(
        residuals,
        np.asarray(start, dtype=np.float64),
        loss=loss,
        f_scale=f_scale_px,
        x_scale="jac",
        max_nfev=int(max_nfev),
        jac_sparsity=_sparsity(
            view_ids, marker_ids, reference_id, len(usable), per_marker
        ),
    )

    corners, _, solved_sizes = unpack(result.x)
    errors = np.linalg.norm(result.fun.reshape(-1, 2), axis=1)
    view_rms = {
        view["name"]: float(np.sqrt(np.mean(errors[rows] ** 2)))
        for view, rows in zip(usable, view_slices)
    }
    layout = {marker_id: corners[index] for index, marker_id in enumerate(all_ids)}
    refined_sizes = {reference_id: float(sizes[reference_id])}
    refined_sizes.update(
        {marker_id: float(size) for marker_id, size in zip(marker_ids, solved_sizes)}
    )
    rms_px = float(np.sqrt(np.mean(errors**2)))
    return layout, view_rms, refined_sizes, rms_px


def _sparsity(view_ids, marker_ids, reference_id, n_views, per_marker):
    """Which parameters each residual can possibly depend on.

    Without this the optimiser probes all of them for every residual, which turns a
    seconds-long fit into a minutes-long one.
    """
    from scipy.sparse import lil_matrix

    marker_column = {marker_id: index for index, marker_id in enumerate(marker_ids)}
    n_marker_params = per_marker * len(marker_ids)
    n_residuals = sum(len(ids) * 8 for ids in view_ids)
    pattern = lil_matrix((n_residuals, n_marker_params + 8 * n_views), dtype=int)

    row = 0
    for view_index, ids in enumerate(view_ids):
        view_columns = slice(
            n_marker_params + 8 * view_index, n_marker_params + 8 * view_index + 8
        )
        for marker_id in ids:
            pattern[row : row + 8, view_columns] = 1
            if marker_id != reference_id:
                start = per_marker * marker_column[marker_id]
                pattern[row : row + 8, start : start + per_marker] = 1
            row += 8
    return pattern


def _implied_sizes(layout):
    """Side length each marker's own corners imply, before any square is enforced."""
    return {
        marker_id: float(
            np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1).mean()
        )
        for marker_id, corners in layout.items()
    }


def _align_to_reference(layout, reference_id, marker_size_mm):
    """Moves the whole layout so the reference marker is upright at the origin.

    The origin is arbitrary — distances never depend on it — but a repeatable one
    makes two calibration runs comparable.
    """
    target = _canonical_square(marker_size_mm)
    source = layout[reference_id]
    source_centroid, target_centroid = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((target - target_centroid).T @ (source - source_centroid))
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1
        rotation = U @ Vt
    return {
        marker_id: (corners - source_centroid) @ rotation.T + target_centroid
        for marker_id, corners in layout.items()
    }


def learn_board(
    views,
    marker_size_mm,
    dictionary,
    intrinsics=None,
    reference_id=None,
    iterations=8,
    ransac_threshold_px=3.0,
    uniform_marker_size=None,
    size_tolerance=0.02,
    refine=True,
):
    """Solves a board's millimetre layout from photos of the bare board.

    Args:
        views (list): Detections, one per photo, as returned by
            `detect.detect_in_images`: dicts with `name` and `corners`.
        marker_size_mm (float): Calipered side of one printed marker, including its
            black border. The only absolute scale in the system — a 1 % error here is
            a 1 % error in every measurement.
        dictionary (int): Predefined ArUco dictionary id, stored on the result.
        intrinsics (CameraIntrinsics, optional): Calibrated lens. Without it the
            layout absorbs the lens distortion and will be slightly wrong; the
            `bootstrap` module exists to break that circle.
        reference_id (int, optional): Marker that anchors the origin. Defaults to the
            one seen in the most photos.
        iterations (int): Alternating passes used to initialise the joint fit. It
            converges in a handful.
        refine (bool): Run the joint optimisation. Leave it on — without it the layout
            keeps a perspective drift that alternating cannot remove (see the module
            docstring). Turning it off avoids the SciPy dependency.
        ransac_threshold_px (float): Inlier threshold when fitting each photo.
        uniform_marker_size (bool, optional): Force every marker to `marker_size_mm`
            (True), let each keep its own measured size (False), or decide from the
            spread of measured sizes (None, the default).
        size_tolerance (float): Relative spread of measured sizes below which markers
            count as uniform.

    Returns:
        BoardSpec: The solved layout, with a populated `diagnostics` dictionary.

    Raises:
        BoardLearningError: If no markers were found, if some marker never shares a
            photo with the rest of the board, or if no photo can be fitted.
    """
    marker_size_mm = float(marker_size_mm)
    if marker_size_mm <= 0:
        raise BoardLearningError(f"Marker size must be positive, got {marker_size_mm}.")

    views = _prepare_views(views, intrinsics)
    reference_id, view_counts = _choose_reference(views, reference_id)
    layout = _seed_layout(views, reference_id, marker_size_mm, ransac_threshold_px)

    # Alternating passes: cheap, robust, and good enough to start the joint fit from.
    scatter, sizes = {}, _implied_sizes(layout)
    for _ in range(int(iterations)):
        layout, scatter = _reestimate(
            _view_homographies(views, layout, ransac_threshold_px), layout
        )
        sizes = _implied_sizes(layout)
        uniform = _is_uniform(sizes, marker_size_mm, uniform_marker_size, size_tolerance)
        layout = {
            marker_id: _fit_square(
                corners, marker_size_mm if uniform else sizes[marker_id]
            )
            for marker_id, corners in layout.items()
        }

    measured = sizes
    uniform = _is_uniform(measured, marker_size_mm, uniform_marker_size, size_tolerance)
    sizes = {marker_id: marker_size_mm for marker_id in layout}
    view_rms = {}

    if refine:
        # Are the markers all one size? Decide it with two short fits from the *same*
        # starting point and the same budget. Starting the second from the first's
        # answer would just hand it the first's progress and it would always look
        # better. A board of one size gains almost nothing from letting the sizes go
        # free, because the extra freedom has nothing real to explain; a board carrying
        # an odd-sized marker improves sharply. That is a far more reliable signal than
        # the spread of a noisy pre-estimate.
        # The rough sizes from the alternating passes gate the comparison rather than
        # settling it. When they agree closely the markers are plainly one size and the
        # second fit is skipped, which is worth doing: proving a uniform board is
        # uniform means running an over-parameterised fit all the way to convergence,
        # and that is by far the slowest thing here.
        uniform = True
        best = _refine_jointly(
            views, layout, sizes, reference_id, max_nfev=_REFINE_BUDGET
        )
        if uniform_marker_size is False or (
            uniform_marker_size is None
            and not _is_uniform(measured, marker_size_mm, None, size_tolerance)
        ):
            mixed = _refine_jointly(
                views, layout, measured, reference_id,
                free_sizes=True, max_nfev=_REFINE_BUDGET,
            )
            if uniform_marker_size is False or mixed[3] < best[3] * (1.0 - _MIXED_SIZE_GAIN):
                best, uniform = mixed, False

        layout, view_rms, sizes, _ = best
        _, scatter = _reestimate(
            _view_homographies(views, layout, ransac_threshold_px), layout
        )
    if not view_rms:
        view_rms = {
            view["name"]: view["rms_px"]
            for view in _view_homographies(views, layout, ransac_threshold_px)
        }

    layout = _align_to_reference(layout, reference_id, marker_size_mm)
    sizes = _implied_sizes(layout)
    size_values = np.array(list(sizes.values()))
    diagnostics = {
        "reference_id": int(reference_id),
        "n_views": len(views),
        "n_markers": len(layout),
        "views_per_marker": {str(i): int(n) for i, n in sorted(view_counts.items())},
        "weakly_constrained_ids": sorted(i for i, n in view_counts.items() if n < 3),
        "marker_scatter_mm": {str(i): round(s, 4) for i, s in sorted(scatter.items())},
        "max_marker_scatter_mm": round(float(max(scatter.values(), default=0.0)), 4),
        "view_rms_px": {name: round(value, 4) for name, value in view_rms.items()},
        "max_view_rms_px": round(float(max(view_rms.values(), default=0.0)), 4),
        "jointly_refined": bool(refine),
        "measured_marker_sizes_mm": {str(i): round(s, 4) for i, s in sorted(sizes.items())},
        "marker_size_spread": round(
            float((size_values.max() - size_values.min()) / marker_size_mm), 5
        ),
        "uniform_marker_size": bool(uniform),
        "lens_corrected": intrinsics is not None,
    }
    diagnostics["warnings"] = _warnings(diagnostics, size_tolerance)

    return BoardSpec(
        dictionary=dictionary,
        marker_size_mm=marker_size_mm,
        corners_mm=layout,
        diagnostics=diagnostics,
    )


def _is_uniform(sizes, marker_size_mm, uniform_marker_size, size_tolerance):
    """Decides whether every marker should be forced to the calipered size."""
    if uniform_marker_size is not None:
        return bool(uniform_marker_size)
    values = np.array(list(sizes.values()))
    return bool((values.max() - values.min()) / marker_size_mm <= size_tolerance)


def _warnings(diagnostics, size_tolerance):
    """Capture problems worth fixing before trusting the layout."""
    warnings = []
    if diagnostics["n_views"] < 8:
        warnings.append(
            f"Only {diagnostics['n_views']} photos used: 15-30 from varied angles and "
            f"distances give a far more stable layout."
        )
    if diagnostics["weakly_constrained_ids"]:
        warnings.append(
            f"Markers {diagnostics['weakly_constrained_ids']} appear in fewer than 3 "
            f"photos, so their positions rest on very little evidence."
        )
    if diagnostics["max_marker_scatter_mm"] > 0.5:
        warnings.append(
            f"Marker positions disagree between photos by up to "
            f"{diagnostics['max_marker_scatter_mm']:.2f} mm: check the board is flat "
            f"and the photos are sharp."
        )
    if diagnostics["max_view_rms_px"] > 2.0:
        warnings.append(
            f"One photo reprojects at {diagnostics['max_view_rms_px']:.2f} px: look for "
            f"motion blur, or a bent board."
        )
    if diagnostics["marker_size_spread"] > size_tolerance:
        warnings.append(
            f"Measured marker sizes vary by {diagnostics['marker_size_spread']:.1%}. "
            f"Either the board carries markers of different sizes, or it is not flat — "
            f"a curled sheet shows up exactly like this."
        )
    if not diagnostics["lens_corrected"]:
        warnings.append(
            "Solved without lens calibration, so this layout has absorbed the lens "
            "distortion. Run garmentiq.calibration.bootstrap to correct it."
        )
    return warnings


def compare_boards(board_a, board_b):
    """Measures how far two layouts of the same board disagree, after alignment.

    Used by `bootstrap` to tell whether a second pass actually changed anything.

    Returns:
        dict: `rms_mm`, `max_mm` and `n_markers` over the markers both boards share.

    Raises:
        BoardLearningError: If the boards share fewer than two markers.
    """
    shared = sorted(set(board_a.corners_mm) & set(board_b.corners_mm))
    if len(shared) < 2:
        raise BoardLearningError(
            f"The two boards share only {len(shared)} markers; nothing to compare."
        )
    points_a = np.vstack([board_a.corners_mm[i] for i in shared])
    points_b = np.vstack([board_b.corners_mm[i] for i in shared])

    # Remove the arbitrary origin and orientation before comparing.
    centroid_a, centroid_b = points_a.mean(axis=0), points_b.mean(axis=0)
    U, _, Vt = np.linalg.svd((points_a - centroid_a).T @ (points_b - centroid_b))
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1
        rotation = U @ Vt
    aligned = (points_b - centroid_b) @ rotation.T + centroid_a

    errors = np.linalg.norm(aligned - points_a, axis=1)
    return {
        "rms_mm": float(math.sqrt(float(np.mean(errors**2)))),
        "max_mm": float(errors.max()),
        "n_markers": len(shared),
    }
