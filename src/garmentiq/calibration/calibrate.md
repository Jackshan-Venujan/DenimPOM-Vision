python -m garmentiq.calibration.camera          # stream works, 640x480?
python -m garmentiq.calibration.chessboard      # board detected?
python -m garmentiq.calibration.capture_lens    # s = save; 15–25 tilted photos covering the frame
python -m garmentiq.calibration.calibrate_lens  # lens error < 0.5 px -> camera_intrinsics.npz
python -m garmentiq.calibration.undistort       # straight lines look straight?
python -m garmentiq.calibration.calibrate_plane # board flat on table, c -> table_homography.npz
python -m garmentiq.calibration.verify_plane    # move board around, v = check
python -m garmentiq.calibration.measure_live    # click a ruler to confirm
