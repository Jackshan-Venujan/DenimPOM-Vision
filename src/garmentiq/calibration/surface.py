# garmentiq/calibration/surface.py
"""Lifting the measuring plane from the board to the top of the fabric.

The homography measures on the board plane, but the landmarks the pipeline detects sit
on the *top surface* of the garment, a few millimetres above it. Anything closer to
the camera looks bigger, so measuring fabric on the board plane overstates every
distance by roughly `offset / camera_height`.

On this rig that is not a rounding error. At 1170 mm with 5 mm of denim::

    5 / 1170 = 0.43 %  ->  4.3 mm on a 1000 mm leg

which is larger than the whole +/-3 mm tolerance. This module rebuilds the homography
one fabric thickness closer to the camera, which needs the camera intrinsics: the
homography alone cannot tell how far away the plane is.
"""
import cv2
import numpy as np


class SurfaceError(ValueError):
    """Raised when the measuring plane cannot be lifted."""


def expected_scale_error(offset_mm, camera_height_mm):
    """Fractional error from measuring a lifted surface on the board plane.

    Args:
        offset_mm (float): Height of the real surface above the board.
        camera_height_mm (float): Perpendicular camera distance to the board.

    Returns:
        float: Relative error, e.g. 0.0043 for 0.43 %. Multiply by a distance to get
        the error in millimetres.
    """
    camera_height_mm = float(camera_height_mm)
    if camera_height_mm <= 0.0:
        raise SurfaceError(f"Camera height must be positive, got {camera_height_mm}.")
    return float(offset_mm) / camera_height_mm


def plane_pose(H, intrinsics):
    """Recovers where the board plane sits relative to the camera.

    The board maps to the image by `K [r1 r2 t]`, so the inverse of the measuring
    homography, pre-multiplied by `K^-1`, holds two rotation columns and the
    translation, all scaled by one unknown factor. Normalising the rotation columns
    recovers that factor, and the third axis is their cross product.

    Args:
        H (numpy.ndarray): (3, 3) pixel -> millimetre homography, from undistorted
            pixels.
        intrinsics (CameraIntrinsics): The calibrated lens.

    Returns:
        tuple:
            - rotation (numpy.ndarray): (3, 3) board frame -> camera frame.
            - translation (numpy.ndarray): (3,) in millimetres.
            - camera_height_mm (float): Perpendicular camera distance to the board.

    Raises:
        SurfaceError: If the homography is not invertible or gives no usable pose.
    """
    H = np.asarray(H, dtype=np.float64)
    try:
        board_to_image = np.linalg.inv(H)
    except np.linalg.LinAlgError as error:
        raise SurfaceError(f"The homography is not invertible: {error}") from error

    M = np.linalg.inv(intrinsics.camera_matrix) @ board_to_image
    norms = np.linalg.norm(M[:, 0]), np.linalg.norm(M[:, 1])
    if min(norms) < 1e-12:
        raise SurfaceError("The homography gives a degenerate plane pose.")
    M = M / np.mean(norms)

    # The board must lie in front of the camera; if it came out behind, flip the sign.
    if M[2, 2] < 0:
        M = -M

    r1, r2 = M[:, 0], M[:, 1]
    r3 = np.cross(r1, r2)
    # Nearest true rotation to the three noisy axes.
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U[:, -1] *= -1
        rotation = U @ Vt

    translation = M[:, 2]
    camera_centre = -rotation.T @ translation
    return rotation, translation, abs(float(camera_centre[2]))


def lift_to_surface(H, intrinsics, offset_mm):
    """Moves the measuring plane `offset_mm` toward the camera.

    Args:
        H (numpy.ndarray): (3, 3) pixel -> millimetre homography on the board plane.
        intrinsics (CameraIntrinsics | None): Calibrated lens. Without it the lift
            cannot be computed and `H` is returned unchanged.
        offset_mm (float): Fabric thickness — how far the measured surface sits above
            the board.

    Returns:
        numpy.ndarray: (3, 3) float64 homography measuring on the lifted surface. The
        millimetre coordinate frame is unchanged, so a point on the board still reads
        the same; only the scale of the plane the pixels are read on has moved.

    Raises:
        SurfaceError: If the plane pose cannot be recovered.
    """
    offset_mm = float(offset_mm)
    if intrinsics is None or offset_mm == 0.0:
        return np.asarray(H, dtype=np.float64).copy()

    rotation, translation, _ = plane_pose(H, intrinsics)
    camera_centre = -rotation.T @ translation
    # Positive Z on the side the camera is on, so the surface rises toward it.
    z = float(np.copysign(offset_mm, camera_centre[2]))

    surface_to_image = intrinsics.camera_matrix @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], z * rotation[:, 2] + translation]
    )
    lifted = np.linalg.inv(surface_to_image)
    return lifted / lifted[2, 2]
