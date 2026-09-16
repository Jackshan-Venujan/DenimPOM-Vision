# garmentiq/calibration/__init__.py
"""Turning pixel positions on the table into millimetres, with a chessboard.

Two calibrations, run once, then measuring:

Step A - lens (again only if the phone, zoom or resolution changes)::

    python -m garmentiq.calibration.camera          # check stream and resolution
    python -m garmentiq.calibration.chessboard      # check the board is detected
    python -m garmentiq.calibration.capture_lens    # save 15-25 tilted board photos
    python -m garmentiq.calibration.calibrate_lens  # -> camera_intrinsics.npz
    python -m garmentiq.calibration.undistort       # straight lines look straight?

Step B - table (again whenever the camera moves)::

    python -m garmentiq.calibration.calibrate_plane # board flat -> table_homography.npz
    python -m garmentiq.calibration.verify_plane    # move board around, check errors
    python -m garmentiq.calibration.measure_live    # click a ruler to confirm

Measuring, in code::

    from garmentiq.calibration.metrology import PlaneMeasurer

    measurer = PlaneMeasurer()
    frame = measurer.undistort(raw_frame)
    length_mm = measurer.distance_mm(point_1, point_2)

All settings live in `config.py`. Modules are imported directly, not re-exported here:
importing a module that is also about to run with `python -m` makes Python execute it
twice.
"""
