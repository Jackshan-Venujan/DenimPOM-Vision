# Chessboard Calibration: Pixels to Millimetres

This document covers the `garmentiq.calibration` package. It describes what the package does, how it works, how to run it, the latest results, and what those results mean for trouser measurement.

---

## Contents

1. [Overview](#1-overview)
2. [Quick start](#2-quick-start)
3. [Hardware and setup](#3-hardware-and-setup)
4. [How it works (techniques)](#4-how-it-works-techniques)
5. [Files and what each one does](#5-files-and-what-each-one-does)
6. [Configuration reference](#6-configuration-reference)
7. [Output files](#7-output-files)
8. [Calibration procedure, step by step](#8-calibration-procedure-step-by-step)
9. [Using the calibration in code](#9-using-the-calibration-in-code)
10. [Quality checks and tolerances](#10-quality-checks-and-tolerances)
11. [Latest results (16 Sep 2026)](#11-latest-results-16-sep-2026)
12. [What the results mean](#12-what-the-results-mean)
13. [Calibration history and lessons learned](#13-calibration-history-and-lessons-learned)
14. [When to recalibrate](#14-when-to-recalibrate)
15. [Troubleshooting](#15-troubleshooting)
16. [Known limitations and next steps](#16-known-limitations-and-next-steps)
17. [Tests](#17-tests)

---

## 1. Overview

The garment models in GarmentIQ work in **pixels**. To judge a trouser measurement against a real tolerance, those pixels have to become **millimetres**. This package provides that conversion for a **fixed phone camera looking at a flat table**.

It is done with two calibrations and one printed chessboard:

| Calibration | Answers the question | Result file |
|---|---|---|
| **A: Lens** | How does this camera's lens bend the image? | `camera_intrinsics.npz` |
| **B: Table** | Which table position, in mm, does each pixel show? | `table_homography.npz` |

After both are done, any pixel on the table can be converted to mm, and so can any distance between two pixels.

```
Step A: lens (once per camera + resolution)
  tilted chessboard photos ──► cv2.calibrateCamera ──► K, dist ──► camera_intrinsics.npz

Step B: table (again whenever the camera moves)
  live frame ──► undistort ──► detect 54 board corners (px)
                                  │  pair with the same corners in mm
                                  ▼
                        cv2.findHomography ──► H ──► table_homography.npz

Step C: measure
  live frame ──► undistort ──► pick pixel points ──► H ──► (x, y) mm ──► distance in mm
```

This package is based on the method in
[OpenCV-Monocular-Metrology](https://github.com/NamitSingh-avi/OpenCV-Monocular-Metrology),
with these changes:

| Original repository | This package |
|---|---|
| 4 corners clicked by hand | 54 corners detected automatically, averaged over 10 frames |
| Units in cm | Units in mm |
| K and dist copied into code by hand | Saved to and loaded from `.npz` files |
| Lens correction not used when measuring | Every frame is undistorted before H is fitted and before measuring |
| One script per task, mixed concerns | One job per module, each runnable on its own |
| USB camera index | DroidCam HTTP stream with a newest-frame reader |

---

## 2. Quick start

Run everything from the repository root, in the `GarmentIQ` environment.

```bash
# Check the setup
python -m garmentiq.calibration.camera          # stream works? resolution 640x480?
python -m garmentiq.calibration.chessboard      # board detected? (54/54 corners)

# Step A: lens (phone in hand, tilted photos)
python -m garmentiq.calibration.capture_lens    # s = save, about 20-40 tilted photos
python -m garmentiq.calibration.calibrate_lens  # -> camera_intrinsics.npz, RMS < 0.5 px
python -m garmentiq.calibration.undistort       # visual check: straight lines straight?

# Step B: table (phone on its fixed mount from here on)
python -m garmentiq.calibration.calibrate_plane # board flat on table, c -> table_homography.npz

# Step C: check accuracy
python -m garmentiq.calibration.verify_plane    # slide board around the table, v = check
python -m garmentiq.calibration.measure_live    # click the ends of a real ruler
```

Each script takes a few seconds to start, because importing `garmentiq` also loads torch.

---

## 3. Hardware and setup

### Camera

| Item | Value |
|---|---|
| Device | Phone, main (1x) camera |
| Streaming app | DroidCam (free version) |
| Stream URL | `http://192.168.8.170:4747/video` |
| Resolution | **640 x 480** (the highest the free version offers) |
| Mounting | Fixed above the table, looking about straight down |

Notes:
- **The resolution is set in the DroidCam app.** `cv2.VideoCapture.set` cannot change it on an HTTP stream.
- **DroidCam accepts one client at a time.** Close any browser tab or other script showing the stream before running a calibration script.
- **Every calibration is valid only at the resolution it was made at.** The scripts refuse to run on frames of a different size.

### Chessboard

| Item | Value |
|---|---|
| Print | `Checkerboard-A3-40mm-9x6` on A3 paper (420 x 297 mm) |
| Squares | **10 x 7** |
| Inner corners (what OpenCV uses) | **9 x 6 = 54** |
| Square size | **40 mm** (nominal; see note) |
| Pattern size | 400 x 280 mm (squares); corner span 320 x 200 mm |
| Mounting | Taped onto a flat, rigid backing |

> **Important:** the "9x6" in the file name counts **inner corners** (where four squares meet), not squares.
> Setting `INNER_CORNERS` to the square count is the most common reason the board is "NOT found".

> **Note on square size:** `SQUARE_MM = 40.0` is the value the print was ordered at. Printers
> scale pages by 1-3 %, and that error carries into **every** measurement. Measure across all 10
> squares with a steel ruler (it should be about 400 mm) and set `SQUARE_MM` to the length divided by 10.

---

## 4. How it works (techniques)

### 4.1 Detecting the chessboard

`cv2.findChessboardCornersSB` (the "sector-based" detector) finds all 54 inner corners with sub-pixel accuracy, using these flags:
- `CALIB_CB_NORMALIZE_IMAGE`: evens out the lighting before detecting.
- `CALIB_CB_ACCURACY`: refines corners on an upsampled image.

It is more accurate and more robust to blur than the older `findChessboardCorners` + `cornerSubPix` combination. It needs **every** inner corner visible. If part of the board is hidden or outside the frame, it finds nothing.

Corners are returned **row by row, left to right**. `board_points_mm()` builds the true positions in the same order:
(0, 0), (40, 0), (80, 0) … (320, 200) mm.

Pairing the two lists row by row is what links pixels to millimetres.

### 4.2 Camera model and lens distortion (Step A)

The camera is modelled as a **pinhole camera** plus **lens distortion**:

- **Camera matrix K**

  ```
  K = | fx   0  cx |     fx, fy: focal length in pixels
      |  0  fy  cy |     cx, cy: image centre (principal point) in pixels
      |  0   0   1 |
  ```

- **Distortion coefficients** `dist = [k1, k2, p1, p2, k3]` (the Brown-Conrady model)
  - `k1, k2, k3`: radial distortion (straight lines bow in or out, strongest at the edges)
  - `p1, p2`: tangential distortion (lens slightly off-centre from the sensor)
  - **`k3` is fixed at 0** (`CALIB_FIX_K3`). With a phone lens and a few dozen views, `k3` is poorly determined and takes wild values that break undistortion near the frame edges.

These values are estimated with **Zhang's method** (`cv2.calibrateCamera`). The same flat board is photographed from many **different, tilted** viewpoints, and the solver finds the K and dist that best explain where all the corners appeared.

> **Why tilt matters:** if every photo sees the board straight-on from the same distance, the
> focal length cannot be separated from the distance. The solver then returns nonsense (see
> [section 13](#13-calibration-history-and-lessons-learned)). Tilted views are what make the focal length measurable.

Quality is measured with the **reprojection error**: the board corners are projected back into each photo with the solved model, and the error is the distance in pixels to where they were actually detected.

### 4.3 Undistortion

`Undistorter` builds the undistortion maps **once** (`cv2.initUndistortRectifyMap`) and applies them to each frame with a fast `cv2.remap`. The corrected image keeps the same K, so it has the same size and scale as the raw frame; only the bending is removed.

For points taken from a **raw** frame, `Undistorter.points()` uses `cv2.undistortPoints` to move them to their undistorted positions.

### 4.4 Table homography (Step B)

A flat surface seen by a pinhole camera relates to the image by a **homography**: one 3 x 3 matrix H with 8 degrees of freedom.

```
| X |       | u |      (u, v) = pixel in the undistorted image
| Y |  ~  H | v |      (X, Y) = position on the table in mm
| 1 |       | 1 |
```

H includes the camera's height, tilt and position, so the result is correct even when the camera is not perfectly straight down.

How H is fitted:
1. **Undistort the frame first**, so H belongs to the undistorted image.
2. **Detect the 54 corners in 10 consecutive frames** and average them, which reduces stream noise. The **jitter** (largest standard deviation of any corner) is reported; above ~0.5 px, the camera or board moved.
3. **Fit H** with `cv2.findHomography(corners_px, corners_mm, 0)`, a plain least-squares fit over all 54 point pairs. With 54 points instead of 4 clicked ones, the fit is much more accurate.
4. **Report the fit error:** each corner is mapped through H and compared with its true mm position.

### 4.5 Measuring

```python
mm_points = cv2.perspectiveTransform(pixel_points, H)   # pixel -> table mm
distance  = ||mm_point_1 - mm_point_2||                  # Euclidean, in mm
```

`mm_per_pixel(point)` gives the local resolution: how many mm one pixel covers at that spot. A landmark that is 1 px off is that many mm off.

### 4.6 Verification

`verify_plane` uses the board itself as a ruler of known size, placed at new spots on the table. Through the saved calibration it measures:
- every single square (should be 40 mm)
- the width span of every row (should be 320 mm)
- the height span of every column (should be 200 mm)
- the diagonal (should be 377.4 mm)

The worst span error is also scaled to an **error per metre** (`error / true length x 1000`) and compared with `TOLERANCE_MM`.

### 4.7 Reading the live stream

A network stream queues frames. If a script is slower than the camera, reading frames in order makes the video fall behind. `Camera` reads in a background thread, keeps only the newest frame, and `read()` never returns the same frame twice.

---

## 5. Files and what each one does

All files are in `src/garmentiq/calibration/`.

| File | Job | Main functions / classes | Run on its own? | Reads | Writes |
|---|---|---|---|---|---|
| `config.py` | Every shared setting in one place | constants only | no | – | – |
| `camera.py` | DroidCam stream reader and window helpers | `Camera`, `check_frame_size`, `open_window`, `put_text` | yes: live view, prints resolution and fps | stream | – |
| `chessboard.py` | Find the board (pixels) and its true corner positions (mm) | `find_corners`, `board_points_mm`, `draw_corners` | yes: live "Board found / NOT found" | stream | – |
| `capture_lens.py` | **Step A1:** save board photos for the lens calibration | `next_image_path` | yes (`s` save, `q` quit) | stream | `lens_images/img_NNN.png` |
| `calibrate_lens.py` | **Step A2:** solve K and dist from the saved photos | `find_corners_in_images`, `solve_intrinsics`, `save_intrinsics` | yes: prints report, PASS/FAIL | `lens_images/` | `camera_intrinsics.npz` |
| `undistort.py` | Remove lens distortion from frames and points | `Undistorter.apply`, `Undistorter.points` | yes: raw vs undistorted side by side (visual only) | `camera_intrinsics.npz` | – |
| `calibrate_plane.py` | **Step B:** fit the table homography H | `average_corners`, `compute_homography`, `save_homography` | yes (`c` calibrate, `q` quit) | stream, intrinsics | `table_homography.npz`, `table_homography.png` |
| `metrology.py` | **The part the rest of GarmentIQ uses:** pixel to mm | `PlaneMeasurer`: `undistort`, `to_mm`, `raw_to_mm`, `distance_mm`, `mm_per_pixel` | no (library) | both `.npz` files | – |
| `verify_plane.py` | **Step C:** accuracy check with the board as a known ruler | `check_board`, `board_outline` | yes (`v` verify, `c` clear, `q` quit) | stream, both `.npz` | – (prints) |
| `measure_live.py` | Click two points and see the distance in mm | `ClickState`, `draw_measurement` | yes (click, `space` freeze, `c` clear, `q` quit) | stream, both `.npz` | – (prints) |
| `__init__.py` | Package docstring with the run order | – | – | – | – |

### Which scripts save something and which only check

| Saves results | Only checks (nothing saved) |
|---|---|
| `capture_lens`, `calibrate_lens`, `calibrate_plane` | `camera`, `chessboard`, `undistort`, `verify_plane`, `measure_live` |

### Import direction

Imports only go one way, so a bug stays in one module:

```
config
  └── camera
        ├── chessboard
        │     ├── capture_lens
        │     └── calibrate_lens
        └── undistort
              ├── calibrate_plane   (+ chessboard)
              └── metrology
                    ├── verify_plane (+ chessboard)
                    └── measure_live
```

Modules are **not** re-exported from `__init__.py`. Importing a module that is also about to run with `python -m` makes Python execute it twice.

---

## 6. Configuration reference

All settings are in `config.py`.

| Setting | Current value | Meaning | Set by |
|---|---|---|---|
| `CAMERA_URL` | `http://192.168.8.170:4747/video` | DroidCam stream address | you |
| `FRAME_SIZE` | `(640, 480)` | Resolution the stream delivers (width, height). Checked on every frame | you (must match the DroidCam app) |
| `DISPLAY_SCALE` | `1` | Window size multiplier for viewing only; clicks are still in frame pixels. Whole numbers only | you |
| `INNER_CORNERS` | `(9, 6)` | Inner corners (columns, rows) = squares - 1 each way | you (count the print) |
| `SQUARE_MM` | `40.0` | Side of one square in mm. **The only real-world scale in the system** | you (measure the print) |
| `OUTPUT_DIR` | `outputs/calibration/` | Where all results go | – |
| `LENS_MIN_IMAGES` | `15` | Warning below this many usable lens photos | – |
| `LENS_MAX_RMS_PX` | `0.5` | PASS limit for lens reprojection error | – |
| `PLANE_AVERAGE_FRAMES` | `10` | Frames averaged before fitting H | – |
| `PLANE_MAX_ERROR_MM` | `0.5` | PASS limit for the homography fit error | – |
| `TOLERANCE_MM` | `3.0` | Allowed garment measurement error (used by `verify_plane`) | project requirement |

### What is calculated automatically and what is typed in

| Calculated automatically and saved (never typed) | Typed in `config.py` (a photo cannot reveal these) |
|---|---|
| K (fx, fy, cx, cy), dist (k1, k2, p1, p2) | `CAMERA_URL`, `FRAME_SIZE` |
| H, including camera height, tilt and position | `INNER_CORNERS`, `SQUARE_MM` |
| Image size and error statistics | limits, paths, `DISPLAY_SCALE` |

Camera distance is **not** a setting. It is inside H, so moving the camera only needs `calibrate_plane` again.

---

## 7. Output files

All in `outputs/calibration/`.

### `lens_images/img_NNN.png`
Raw 640 x 480 frames saved by `capture_lens`, each with the board fully detected. `calibrate_lens` uses **all** of them.

> Delete this folder before recalibrating a different camera, zoom or resolution; otherwise old and new photos are mixed.

### `camera_intrinsics.npz`

| Key | Meaning |
|---|---|
| `K` | 3 x 3 camera matrix |
| `dist` | `[k1, k2, p1, p2, k3]` (k3 = 0) |
| `image_size` | `[width, height]` it is valid for |
| `rms_px` | overall reprojection error |
| `per_view_rms_px` | error of each photo |

### `table_homography.npz`

| Key | Meaning |
|---|---|
| `H` | 3 x 3 homography, undistorted pixel to table mm |
| `image_size` | `[width, height]` it is valid for |
| `rms_error_mm`, `max_error_mm` | fit error on the board corners |
| `square_mm`, `inner_corners` | board settings used |

### `table_homography.png`
The undistorted frame with the averaged corners drawn, as a record of where the board was when H was fitted.

---

## 8. Calibration procedure, step by step

### Step 0: Check the setup

1. Start DroidCam on the phone and set the resolution. Stay on the **1x lens** with no zoom.
2. `python -m garmentiq.calibration.camera`: check it shows `640x480` and a live picture.
3. `python -m garmentiq.calibration.chessboard`: check you see **"Board found: 54/54 corners"**.

### Step A: Lens calibration

**Prepare:**
- Delete old photos in `outputs/calibration/lens_images/`.
- Tape the board onto a rigid, flat backing.
- Use bright, even light with no glare on the squares.

**Capture** with `python -m garmentiq.calibration.capture_lens`. The easiest way is to keep the board flat on the table and **hold the phone in your hand**, moving it around.

Each photo should combine all three kinds of movement:

| Movement | What to do |
|---|---|
| **Distance** | Some photos close (board ~60% of the frame), some medium (~40%), some far (~20-25%) |
| **Position and rotation** | Board in the centre, each corner and each edge of the frame; phone turned 0°, 45° and 90° |
| **Tilt (essential)** | Phone leaning **20-45°** forward, back, left, right and diagonally, so the board looks like a trapezoid |

```
   PARALLEL (straight down)    TILTED (~20-45°)          PERPENDICULAR (edge-on)
          [phone]                    [phone]
             |                          \                  [phone] ----->
             v                           \
   ====== board ======         ====== board ======         ====== board ======
   OK for only 1-3 photos      MOST photos like this       never (board not visible)
```

At each shot, wait for "Board found", **hold still for about 1 second**, then press `s`. The yellow dots show coverage and should reach the whole frame.

Avoid:
- only straight-down photos
- touching or bending the board
- angles steeper than 60°
- motion blur
- changing zoom, lens or resolution mid-way

**Solve** with `python -m garmentiq.calibration.calibrate_lens`, then check:

| Value | Good result |
|---|---|
| Overall RMS | < 0.5 px |
| fx, fy | nearly equal, roughly 450-700 px at 640 x 480 |
| cx, cy | near 320, 240 |
| k1, k2 | small, roughly between -1 and +1 |

Delete photos marked `<- outlier` and run it again if needed.

**Look** with `python -m garmentiq.calibration.undistort` (optional). Straight lines near the edges should look straight in the right-hand view.

### Step B: Table calibration

1. **Mount the phone in its final position**, looking about straight down. **Don't move it after this.**
2. Lay the board **flat on the table**, in the middle of the garment area. Don't prop it up; it must lie on the same surface as the garment.
3. `python -m garmentiq.calibration.calibrate_plane`. When "Board found" shows, press `c` and keep still while 10 frames are averaged.
4. Check `Fit error ... -> PASS` and a jitter below 0.5 px. Pressing `c` again overwrites the file with the new fit.

### Step C: Verify

1. `python -m garmentiq.calibration.verify_plane`: slide the board (flat) to the centre, the corners of the garment area and the frame edges. Press `v` at each spot.
2. `python -m garmentiq.calibration.measure_live`: lay a **long steel ruler or tape** (ideally 1 m) on the table, press `space` to freeze, click both ends, and compare with the true length. Repeat at a few spots.

### Camera position rules per script

| Script | Camera height and angle | Board |
|---|---|---|
| `camera`, `chessboard`, `undistort` | doesn't matter | – |
| `capture_lens` | **vary it** (height and tilt) | flat on the table |
| `calibrate_lens` | no camera used | – |
| `calibrate_plane` | **fixed final position** | flat on the table |
| `verify_plane`, `measure_live`, garment measuring | **same fixed position** | flat on the table |

---

## 9. Using the calibration in code

```python
from garmentiq.calibration.metrology import PlaneMeasurer

measurer = PlaneMeasurer()                   # loads both .npz files from outputs/calibration/

frame = measurer.undistort(raw_frame)        # 1. undistort the camera frame
points_mm = measurer.to_mm([(120, 200), (400, 210)])       # 2a. undistorted pixels -> mm
length_mm = measurer.distance_mm((120, 200), (400, 210))   # 2b. distance in mm

# If points come from the RAW (not undistorted) frame, e.g. landmarks detected on it:
points_mm = measurer.raw_to_mm(raw_landmark_points)

# Local resolution: mm covered by one pixel at a spot
scale = measurer.mm_per_pixel((320, 240))
```

### In the real-time pipeline (`realtime_measure.py`)

The live trouser pipeline uses the calibration like this:

1. **Load the calibration first.** `PlaneMeasurer()` is loaded at start-up, before the models, so a missing calibration fails immediately.
2. **Undistort every frame.** `read_undistorted()` reads the newest frame from the shared `Camera` (same `CAMERA_URL` as the calibration) and undistorts it. Classification, segmentation and landmark detection all run on this undistorted frame, so their landmark pixels can be converted to mm directly.
3. **Frames are never resized.** H belongs to the calibrated resolution; the models resize their own inputs internally.
4. **Convert measurements to mm.** After landmark detection, refinement and derivation, `measurements_in_mm()` takes each measurement's start and end landmark, maps **both points** onto the table with H, and measures the distance in mm. This handles perspective correctly, unlike multiplying the pixel distance by one mm-per-pixel factor. For trousers, **"full length"** runs from landmark 1 (waist left) to landmark 6 (hem left outer).
5. **Output.**
   - The review window and console show mm; the console also shows px.
   - The saved `outputs/realtime/*_measurement.json` keeps `distance` (px) and adds `distance_mm` for each measurement.

Rules:
- Points must come from a frame at the **calibrated resolution** (640 x 480).
- Use `to_mm` / `distance_mm` for points from the **undistorted** frame, and `raw_to_mm` for points from the **raw** frame.
- Only points **on the table surface** measure correctly. Anything higher (thick seams, folds) is closer to the camera and measures slightly long.

---

## 10. Quality checks and tolerances

| Check | Where | Limit | What it tells you |
|---|---|---|---|
| Lens reprojection RMS | `calibrate_lens` | < **0.5 px** | How well K and dist explain the photos |
| Per-photo error | `calibrate_lens` | outlier if > 2 x median (and > 0.5 px) | Blurred or bent-board photos to delete |
| Plausibility of K, dist | `calibrate_lens` (read by eye) | fx ≈ fy ≈ 450-700 px, \|k1\|, \|k2\| < ~1 | Catches degenerate calibrations that still "fit" |
| Corner jitter | `calibrate_plane` | < **0.5 px** | Camera or board moved during averaging |
| Homography fit error | `calibrate_plane` | max < **0.5 mm** | How well H fits the board corners |
| Error per metre | `verify_plane` | ≤ **3.0 mm** (`TOLERANCE_MM`) | Span error scaled to 1 m length |
| Real ruler | `measure_live` | within **3.0 mm** | The final end-to-end check |

**Project tolerance: 3 mm** on garment measurements, with a fixed camera and the garment flat on the table.

---

## 11. Latest results (16 Sep 2026)

### 11.1 Lens calibration: PASS

37 photos, 640 x 480, tilted views.

| Value | Result |
|---|---|
| fx, fy | **499.06, 498.91 px** (0.03 % apart) |
| cx, cy | **304.7, 231.3 px** |
| k1, k2 | **0.1335, -0.2531** |
| p1, p2 | **-0.0052, -0.0045** |
| k3 | 0 (fixed) |
| Field of view | 65.3° horizontal, 51.4° vertical |
| Overall RMS | **0.193 px** (limit 0.5) |
| Per-photo error | median 0.165 px, worst 0.457 px |

```
K = | 499.06     0    304.75 |        dist = [0.1335, -0.2531, -0.0052, -0.0045, 0]
    |    0    498.91  231.35 |
    |    0       0      1    |
```

### 11.2 Table homography: PASS (saved fit)

| Value | Result |
|---|---|
| Frames averaged | 10 (jitter 0.02 px) |
| Scale on board | **2.74 mm per pixel** |
| Fit error | RMS **0.305 mm**, max **0.493 mm** (limit 0.5), which is 0.11 / 0.18 px |
| Camera height above table (derived from K and H) | **about 1390 mm** |
| Camera tilt from straight down (derived) | **about 5°** |
| Table area covered by the frame | about **1720-1840 mm x 1300 mm** |
| Scale across the frame (`mm_per_pixel`) | 2.57 mm/px (bottom-left) to 3.04 mm/px (top-right), 2.79 at the centre |

Four fits were run. Two failed with max errors of 0.75 mm, which is only 0.27 px. At this resolution the 0.5 mm limit is stricter than one pixel (2.74 mm), so that result is noise, not a real problem. The saved fit is the last one, which passed.

### 11.3 Verification with the board (`verify_plane`)

The true spans are width 320 mm, height 200 mm and diagonal 377.4 mm. Errors are measured minus true.

| Check | Squares mean / max | Width | Height | Diagonal | Per metre | Result |
|---|---|---|---|---|---|---|
| #1 | 0.25 / 0.67 mm | -1.41 | -0.66 | -1.44 | 4.39 | FAIL |
| #2 | 0.21 / 0.74 mm | +1.56 | +1.03 | +1.94 | 5.16 | FAIL |
| #3 | 0.26 / 1.44 mm | +2.04 | +1.38 | +2.31 | 6.88 | FAIL |
| #4 | 0.29 / 1.07 mm | +1.09 | +1.58 | +3.02 | 8.00 | FAIL |
| #5 | 0.32 / 2.08 mm | +1.79 | +1.94 | +1.89 | 9.72 | FAIL |
| #6 | 0.21 / 1.13 mm | -0.78 | +1.19 | +3.63 | 9.63 | FAIL |

- **Span errors:** 0.7-3.6 mm, which is about **0.3-1.3 px**.
- **Mean single-square error:** 0.21-0.32 mm.

### 11.4 Measuring by hand (`measure_live`)

| Clicked pixels | Distance |
|---|---|
| (199, 114) → (474, 130) | 781.0 mm |
| (200, 116) → (473, 132) | 775.1 mm |
| (192, 175) → (472, 181) | 787.2 mm |
| (300, 245) → (312, 248) | 34.3 mm |

The first two measurements are the **same line, clicked twice**. Each end moved by only 1-2 px, and the result changed by **5.9 mm**.

---

## 12. What the results mean

### The calibration itself is good
- **Lens:** fx ≈ fy, the centre is near the middle of the image, distortion is small, and the RMS is 0.19 px. This is a healthy, physically plausible phone-camera calibration.
- **Homography:** it fits the 54 corners to within 0.18 px, and the camera height (~1390 mm) derived from K and H matches the measured scale (1390 / 499 ≈ 2.78 mm/px).

### Accuracy is limited by resolution, not by the calibration

At 640 x 480, **one pixel covers 2.6-3.0 mm of table**. So:
- A landmark or click that is **1 px off** moves its end by **~2.7 mm**.
- A length has two ends, so a single pixel of error at both ends gives **~4-5.5 mm**. `measure_live` showed exactly this (5.9 mm for a 1-2 px change).
- The `verify_plane` span errors are all about 1 px or less, and their signs are mixed. That is **random pixel noise**, not a systematic calibration error.

**Conclusion: the 3 mm tolerance cannot be met at 640 x 480**, however good the calibration is.

### About the "per metre" numbers

The per-metre value assumes the error grows with length. That is true for scale errors, but **not** for pixel noise at the two ends of a span. For example, a 1 px (2.7 mm) error on the 200 mm board height counts as ~14 mm per metre. At this resolution the per-metre check therefore **overstates** the error, so its FAILs should not be read literally. A long real ruler in `measure_live` is the better test.

### Expected accuracy at other resolutions (same camera height)

| Resolution | mm per pixel | Error from 1 px at both ends | Meets 3 mm? |
|---|---|---|---|
| 640 x 480 (now) | ~2.7 | ~4-5.5 mm | **no** |
| 1280 x 720 | ~1.4 | ~2-3 mm | borderline |
| 1920 x 1080 | ~0.9 | ~1.3-1.8 mm | yes |
| Full phone photo (~4600 px wide) | ~0.4 | < 1 mm | comfortably |

---

## 13. Calibration history and lessons learned

| # | What happened | Cause | Fix / lesson |
|---|---|---|---|
| 1 | Board "NOT found" with the printed board partly under a second (green) chessboard | The detector needs **all** inner corners visible; two boards in view | Only one board in view, fully visible, with a white border |
| 2 | Board still "NOT found" when fully visible | `INNER_CORNERS` was `(8, 5)`: "9x6" was wrongly read as squares | Set to `(9, 6)`. **Count inner corners = squares - 1** |
| 3 | First lens calibration: fx = 4447, fy = 5019, k2 = -3742 | Most likely the same as #4: board flat on the table and seen straight-on (photos not inspected) | Focal length can't be measured without tilt |
| 4 | Second lens calibration: fx = 12 326, fy = 14 174, k2 = -98 773, RMS 1.31 px | 21 photos all at the same distance (board 23-26 % of frame) and straight-on (opposite edges within 3 % of each other). One photo had a hand bending the board (5.35 px error) | **Tilt the phone 20-45°**, vary the distance, keep hands off the board, use a rigid backing |
| 5 | Final lens calibration: fx = fy = 499, RMS 0.19 px | 37 tilted photos at varied distances | **PASS** |
| 6 | Homography "FAIL" at 0.51-0.75 mm | Limit (0.5 mm = 0.18 px) is stricter than the resolution supports | Not a real problem at 640 x 480; the saved fit passed |
| 7 | `verify_plane` FAILs of 4-10 mm per metre | ~1 px corner noise at 2.7 mm/px, amplified by per-metre scaling | Resolution is the limit (section 12) |

---

## 14. When to recalibrate

| You change... | Redo |
|---|---|
| Camera height, angle or position, or the table | `calibrate_plane` → `verify_plane` |
| Phone, lens (1x / wide), zoom or **resolution** | Delete `lens_images/` → `capture_lens` → `calibrate_lens` → `calibrate_plane` → `verify_plane` (and update `FRAME_SIZE` if the resolution changed) |
| Printed board | Update `INNER_CORNERS` / `SQUARE_MM` → redo both calibrations |
| `SQUARE_MM` after measuring the print | `calibrate_plane` (the lens calibration's pixel results don't depend on it) |
| Nothing, but results look off | `verify_plane` and `measure_live` with a ruler first |

---

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Could not open camera stream` | DroidCam not running, wrong IP, or another client connected | Start DroidCam, check the IP and Wi-Fi, close browser tabs and other scripts |
| `No frame from the camera for 5 s` | Stream stalled | Restart DroidCam |
| `Frame is WxH but ... is 640x480` | Resolution changed in the app | Set it back, or recalibrate at the new resolution |
| "Board NOT found" | Wrong `INNER_CORNERS`, part of the board hidden or off-frame, glare, second board in view | Count squares - 1, show the whole board with a white border, remove glare |
| fx, fy in the thousands; huge k1 or k2 | Lens photos not tilted, or all at one distance | Retake tilted photos at varied distances (section 8) |
| One photo marked outlier | Blur, bent board, hand on board | Delete that photo and run `calibrate_lens` again |
| Jitter > 0.5 px in `calibrate_plane` | Camera or board moved, vibration | Keep still, check the mount |
| Measurements off everywhere after moving the phone | H no longer matches the camera position | Run `calibrate_plane` again |
| `... not found. Run ... first` | Missing `.npz` file | Run the earlier step |
| Clicks land in the wrong place | Window scaling issue | Check the printed `Clicked pixel` values stay within 0-639 and 0-479 |

---

## 16. Known limitations and next steps

1. **Resolution (main limit).** At 640 x 480 the error from one pixel alone exceeds 3 mm. Options:
   - DroidCam Pro at 1920 x 1080
   - IP Webcam app (free, higher resolutions)
   - Full-resolution phone photos for the measurement itself

   Changing the resolution means redoing both calibrations.
2. **`SQUARE_MM` not yet measured.** It is still the nominal 40.0 mm. Measure the print (section 3) and rerun `calibrate_plane`.
3. **Fabric height.** H measures the table surface, but the fabric's top surface is a few mm higher, so measurements come out slightly long. A 5 mm offset at ~1390 mm camera height is about 0.36 %, or ~4.3 mm over a 1200 mm trouser. This is not corrected yet.
4. **Camera tilt about 5°.** The scale varies from 2.57 to 3.04 mm/px across the frame. H corrects for this, but resolution is coarser toward the top-right. Mounting the camera closer to straight down evens this out.
5. **Board smaller than the garment.** H is fitted on a 320 x 200 mm corner span and used over about 1200 x 700 mm. If long-ruler tests at higher resolution show errors growing toward the edges, possible upgrades are:
   - fitting H from K plus a `solvePnP` pose (6 unknowns instead of 8)
   - combining several board placements
6. **`verify_plane` per-metre metric.** It overstates random pixel noise at low resolution (section 12). A possible improvement is to estimate one scale factor from all 54 corners together.
7. **Only `realtime_measure.py` reports mm so far.** `full_pipeline.ipynb` still reports pixels. It works on saved photos, which would have to be taken with the calibrated camera, position and resolution, and converted with `PlaneMeasurer.raw_to_mm`.
8. **No automatic plausibility check.** `calibrate_lens` doesn't yet warn about impossible values (for example fx in the thousands); they have to be read by eye.

---

## 17. Tests

`test/test_calibration.py` has 12 tests on **synthetic** data: a virtual camera photographs a virtual board, so the correct answers are exactly known. No camera or network is needed.

| Test class | Checks |
|---|---|
| `TestChessboard` | Corner order and mm positions; detection on a rendered board; `None` without a board |
| `TestCalibrateLens` | Recovers the known focal length from 10 tilted synthetic views |
| `TestCalibratePlane` | H maps a tilted board back to mm exactly; averaging reduces noise and reports jitter |
| `TestMetrology` | Distance and mm-per-pixel; real homography measures the board; rejects mismatched resolutions and wrong frame sizes |
| `TestVerifyPlane` | Exact calibration passes; a 1 % scale error is reported as 10 mm per metre and fails |

Run:

```bash
python -m pytest test/test_calibration.py -o addopts="" -v
```

(`-o addopts=""` skips the repository's default `--nbmake` option, which is only needed for notebooks.)

