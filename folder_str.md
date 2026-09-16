# GarmentIQ Project Structure and Development Guide

## 1. What this project does

GarmentIQ turns a garment photograph into measurements by combining several computer-vision stages:

```text
Input image
    -> garment classification
    -> garment segmentation
    -> landmark detection
    -> landmark refinement
    -> derived landmark calculation
    -> pixel-distance measurements
    -> JSON, CSV, masks, and annotated images
```

The stages are modular. You can run the complete workflow through `tailor.py`, or run each stage separately while developing and debugging.

---

## 2. Root-level files and folders

### `full_pipeline.ipynb`

A local end-to-end notebook for this project. It uses the models in `models/`, reads images from `test_images/`, and writes results to `outputs/trouser_measurements/`.

It demonstrates:

1. Importing the package and models.
2. Classifying the garment.
3. Segmenting the garment.
4. Detecting landmarks.
5. Refining landmarks with the mask.
6. Deriving additional landmarks.
7. Computing pixel measurements.
8. Running the complete `tailor` pipeline.

Use this notebook as the main local experiment while developing the trousers workflow.

### `README.md`

The project overview and links to the original tutorials, documentation, model information, and acknowledgements.

### `pyproject.toml`

The Python project configuration. It defines package metadata, dependencies, and build settings used when installing or synchronizing the project.

### `LICENSE`

The project license.

### `src/`

The source code for the installable `garmentiq` Python package.

### `models/`

Local model weights used by the notebooks and pipeline:

- `tiny_vit_inditex_finetuned.pt`: garment classification weights.
- `hrnet.pth`: HRNet garment landmark detection weights.
- `birefnet/model.safetensors`: BiRefNet segmentation weights.
- `sam/`: optional SAM model assets.

Model architecture code lives under `src/garmentiq`; the learned weights live here.

### `test_images/`

Local input photographs used for experiments. Keep test images organized by garment type when comparing results.


### `output/` and `outputs/`

Generated results. Depending on which notebook or pipeline was run, these folders can contain:

- `metadata.csv`: one row per processed image.
- `mask_image/`: saved segmentation masks.
- `measurement_image/`: images annotated with landmarks and measurements.
- `measurement_json/`: structured measurement results.

Avoid treating generated files as source code. They can be regenerated from the input image, model weights, and configuration.

### `src/garmentiq.egg-info/`

Packaging metadata generated during installation or building. It is not application logic and normally should not be edited manually.

---

## 3. The `src/garmentiq` package

### `__init__.py`

The public package entry point. It allows code such as:

```python
import garmentiq as giq
```

It exposes `classification`, `segmentation`, `landmark`, `matting`, `utils`, and `tailor`. The optional `grounding` package is loaded lazily because it has heavier optional dependencies.

### `tailor.py`

The complete pipeline coordinator. The `tailor` class loads the configured models and processes a folder of images through classification, segmentation, optional matting, landmark detection, refinement, derivation, and measurement.

Use it after the individual stages are working reliably. Use the individual modules first when debugging or experimenting.

---

## 4. `classification/`

This module identifies the garment category. The category controls which landmark range and measurement instruction file are used later.

Files:

- `__init__.py`: public classification API.
- `model_definition.py`: model architectures such as `CNN3`, `CNN4`, and `tinyViT`.
- `load_model.py`: creates a model and loads checkpoint weights.
- `predict.py`: preprocesses an image and returns the predicted class and probabilities.
- `load_data.py`: loads image data for training or evaluation.
- `train_test_split.py`: divides data into training and test sets.
- `train_pytorch_nn.py`: trains a classifier from scratch.
- `fine_tune_pytorch_nn.py`: adapts a pretrained classifier to another dataset.
- `test_pytorch_nn.py`: evaluates classifier performance.
- `utils.py`: dataset, training, validation, and checkpoint helpers.

Typical output:

```python
label, probabilities = giq.classification.predict(...)
```

Example labels include `trousers`, `shorts`, `skirt`, `vest`, and several top and dress categories.

---

## 5. `segmentation/`

This module separates the garment from the background and produces a mask. The mask is required by landmark refinement and landmark derivation.

Files:

- `__init__.py`: public segmentation API.
- `load_model.py`: common model-loading entry point.
- `extract.py`: runs segmentation and returns the original image and mask.
- `plot.py`: displays or plots a mask.
- `change_background_color.py`: composites the garment onto another background color.
- `process_and_save_images.py`: batch processing and saving helpers.
- `model_definition/`: segmentation model implementations.
- `model_definition/birefnet/`: BiRefNet architecture and configuration.
- `model_definition/sam/`: SAM-related implementations and processors.

Supported approaches depend on the installed models and configuration. BiRefNet does not need a prompt; SAM can use point, label, box, or text-guided prompts.

Typical output:

```python
original_image, mask = giq.segmentation.extract(...)
```

---

## 6. `landmark/`

This module manages the points used for measurements.

A landmark is a meaningful garment location, for example:

- left and right waist points
- shoulder points
- sleeve ends
- hip points
- trouser hems
- center points
- garment boundary intersections

Top-level files:

- `__init__.py`: public landmark API.
- `detect.py`: predicts predefined landmarks with the HRNet model.
- `refine.py`: adjusts predicted points using the segmentation mask.
- `derive.py`: calculates landmarks that the model does not directly predict.
- `plot.py`: draws landmarks on images.
- `utils.py`: landmark lookup and detection-dictionary helpers.

### `landmark/detection/`

The HRNet landmark detector implementation:

- `model_definition.py`: HRNet architecture.
- `load_model.py`: loads HRNet weights.
- `utils.py`: detector preprocessing and output helpers.
- `__init__.py`: detection submodule exports.

### `landmark/refinement/`

The refinement implementation:

- `refine_landmark_with_blur.py`: searches around a predicted point on a blurred mask to move it toward the garment boundary.
- `__init__.py`: refinement exports.

Refinement does not teach the model a new point. It improves the location of an existing predicted point.

### `landmark/derivation/`

The geometric derivation implementation:

- `derivation_dict.py`: derivation rules for garment classes and landmarks.
- `derive_keypoint_coord.py`: calculates a derived point.
- `line_intersect.py`: line-intersection geometry.
- `mask_intersect.py`: finds intersections with the segmentation mask.
- `prepare_args.py`: prepares arguments for derivation rules.
- `process.py`: coordinates derivation operations.
- `utils.py`: derivation helpers.
- `__init__.py`: derivation exports.

Derivation is the preferred extension point when a new measurement point can be computed reliably from existing landmarks and the segmentation mask.

---

## 7. `garment_classes.py`

This file is the registry of supported garment classes. Each class includes information such as:

- how many predefined points it has
- which part of the HRNet output belongs to that garment
- which instruction JSON defines its measurements

For example, the trousers entry contains 14 predefined points and points to `instruction/trousers.json`.

The class registry and instruction JSON must agree. If the garment name or point structure is changed in one place, check the other place too.

---

## 8. `instruction/`

This folder contains garment-specific JSON schemas:

- `long sleeve dress.json`
- `long sleeve top.json`
- `short sleeve dress.json`
- `short sleeve top.json`
- `shorts.json`
- `skirt.json`
- `trousers.json`
- `vest dress.json`
- `vest.json`

An instruction file describes landmarks and measurements. A measurement normally specifies two landmark IDs: a start point and an end point.

The instruction files are configuration, not neural-network code. They are the first place to edit when adding a measurement between points that already exist.

---

## 9. `matting/`

Matting creates a soft alpha mask instead of a hard foreground/background mask. It is useful when edges, loose fibers, lace, or semi-transparent details need more natural compositing.

Files:

- `__init__.py`: public matting API.
- `load_model.py`: loads the matting model.
- `trimap.py`: creates a trimap from a segmentation mask.
- `matte.py`: generates the alpha matte and composites it with a background.
- `model_definition/`: ViTMatte and Matting Anything model definitions.

Matting is optional. Landmark refinement and derivation still fundamentally depend on a usable garment mask.

---

## 10. `grounding/`

This optional module uses Grounding DINO to convert a text phrase into image boxes. The boxes can then be passed to text-guided or box-guided segmentation.

Files:

- `__init__.py`: grounding exports.
- `grounding_dino.py`: loads Grounding DINO configuration and processor, then converts text into regions.

Example concept:

```text
text prompt: "trousers"
    -> Grounding DINO bounding box
    -> SAM segmentation prompt
    -> garment mask
```

This module is useful when images contain multiple objects or when segmentation needs an explicit object prompt.

---

## 11. `measurement/`

This folder is currently empty in this project copy. The current measurement-distance implementation is in:

```text
src/garmentiq/utils/compute_measurement_distances.py
```

That function calculates Euclidean distances between landmark pairs in pixels and adds those distances to the detection dictionary.

A future development could move measurement logic into this folder, especially if the project gains multiple measurement systems, calibration, perspective correction, or uncertainty handling.

---

## 12. `utils/`

Shared helpers used by multiple pipeline stages:

- `compute_measurement_distances.py`: computes Euclidean landmark distances.
- `device.py`: validates and resolves CPU, CUDA, or MPS devices.
- `checkpoint.py`: checkpoint loading and validation helpers.
- `clean_detection_dict.py`: cleans or normalizes detection data.
- `validate_garment_class_dict.py`: validates garment class configuration.
- `check_filenames_metadata.py`: checks image names and metadata.
- `check_unzipped_dir.py`: validates extracted folders.
- `export_dict_to_json.py`: writes dictionaries as JSON.
- `unzip.py`: archive extraction helper.
- `__init__.py`: utility exports.

---

## 13. What the notebooks in `test/` are for

The notebooks are examples and experiments, not all unit tests.

### `tutorial_classification.ipynb`

Shows how to load the classifier, classify an image, and inspect class probabilities.

Use it when:

- checking whether an image is recognized as the right garment
- testing a new classifier
- measuring classification accuracy

### `tutorial_segmentation.ipynb`

Shows how to create a garment mask with BiRefNet or SAM.

Use it when:

- checking whether the garment boundary is accurate
- comparing segmentation models
- experimenting with point, box, label, or text prompts

### `tutorial_grounding.ipynb`

Shows how a text prompt is converted into a bounding box with Grounding DINO for use by segmentation.

Use it when:

- an image contains multiple objects
- the segmentation model needs a box or text-guided prompt

### `tutorial_matting.ipynb`

Shows how to refine a hard mask into a soft alpha matte.

Use it when:

- edges need to look natural
- the output will be composited onto another background
- fabric details need softer boundaries

### `tutorial_landmark_detection.ipynb`

Shows how HRNet predicts garment landmarks.

Use it when:

- checking landmark positions
- checking confidence values
- understanding which points the model predicts directly

### `tutorial_landmark_refinement_and_derivation.ipynb`

Shows two important operations:

- refinement moves detected landmarks closer to the actual garment boundary
- derivation creates new landmarks using geometric rules and the segmentation mask

This is the most useful tutorial for custom points of measure.

### `tutorial_tailor.ipynb`

Shows the complete folder-based pipeline. Use it as the reference for processing many images and saving masks, annotated images, measurement JSON, and metadata.

### `adv_usage_custom_measurement_instruction.ipynb`

Shows how to define custom garment instructions and measurements. Start here before changing model architecture.

### `adv_usage_classification_model_training_evaluation.ipynb`

Shows how to train and evaluate classification models.

### `adv_usage_classification_model_fine_tuning.ipynb`

Shows how to fine-tune the pretrained `tinyViT` classifier for a different catalog or dataset.

---

## 14. Adding a custom measurement

There are three levels of customization.

### Level 1: Add a measurement between existing landmarks

This is the easiest option and normally requires no model retraining.

1. Open the relevant instruction file, such as `src/garmentiq/instruction/trousers.json`.
2. Add a new measurement entry.
3. Set its start and end landmark IDs to existing landmarks.
4. Run the pipeline.
5. Plot the two points and verify the result on several images.

Conceptual example:

```json
"hip_width": {
  "landmarks": {
    "start": "left_hip",
    "end": "right_hip"
  }
}
```

The exact landmark IDs must match the IDs already used by the instruction schema.

### Level 2: Add a geometrically derived point

Use this when the model does not predict the point but it can be calculated from reliable points and the mask.

Recommended process:

1. Define the new landmark in the instruction JSON.
2. Mark it as derived or non-predefined according to the existing schema.
3. Add or reuse a rule in `landmark/derivation/derivation_dict.py`.
4. Use line, midpoint, or mask-intersection geometry.
5. Run `giq.landmark.derive(...)`.
6. Plot the derived point on the original image.
7. Add a measurement that uses the new point.
8. Test on many garment shapes, poses, colors, and backgrounds.

Good derived points include boundary intersections, midpoints, and points at a known geometric relationship to existing landmarks.

### Level 3: Teach the landmark model a new point

Retrain or fine-tune HRNet only when geometry cannot reliably determine the point.

This requires:

1. Annotated training images containing the new point.
2. A consistent point definition.
3. Changes to the landmark output layout.
4. Changes to garment class point counts and index ranges.
5. A new or fine-tuned HRNet checkpoint.
6. Updates to the instruction schema.
7. Evaluation of point accuracy and confidence.

This has a larger impact than editing an instruction file or adding a derivation rule. Keep the existing point ordering stable when possible, because changing indexes can affect every garment class using the shared model output.

---

## 15. Pixel measurements versus physical measurements

The current distance calculation returns pixel distances:

```text
distance_pixels = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

A value such as `250` currently means approximately 250 pixels. It does not automatically mean 250 cm or 250 mm.

For physical measurements, add calibration. Possible references include:

- a ruler in the image
- an ArUco marker with a known size
- a known garment dimension
- a controlled camera and capture distance

A simple scale conversion is:

```text
physical_distance = pixel_distance / pixels_per_unit
```

For better accuracy, also consider perspective correction, garment orientation, camera distance, and whether the garment is laid flat.

The existing `input_with_marker/` folder can be useful for developing a marker-based calibration stage.

---

## 16. Building a complete pipeline from separate modules

Load the models once, then process each image through the stages in order.

```python
from pathlib import Path

import garmentiq as giq
from garmentiq.classification.model_definition import tinyViT
from garmentiq.garment_classes import garment_classes
from garmentiq.landmark.derivation.derivation_dict import derivation_dict
from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
from garmentiq.segmentation.model_definition.birefnet import (
    BiRefNet,
    load_birefnet_config,
)

ROOT = Path.cwd()
IMAGE_PATH = ROOT / "test_images" / "trouser.jpeg"
MODEL_DIR = ROOT / "models"
DEVICE = "cuda"

classifier = giq.classification.load_model(
    model_path=str(MODEL_DIR / "tiny_vit_inditex_finetuned.pt"),
    model_class=tinyViT,
    model_args={"num_classes": 9, "img_size": (120, 184), "patch_size": 6},
    device=DEVICE,
)

segmenter = giq.segmentation.load_model(
    model_path=str(MODEL_DIR / "birefnet" / "model.safetensors"),
    model_class=BiRefNet,
    model_args=load_birefnet_config(),
    device=DEVICE,
)

landmark_model = giq.landmark.detection.load_model(
    model_path=str(MODEL_DIR / "hrnet.pth"),
    model_class=PoseHighResolutionNet(),
    device=DEVICE,
)

label, probabilities = giq.classification.predict(
    model=classifier,
    image_path=str(IMAGE_PATH),
    classes=sorted(garment_classes),
    resize_dim=(120, 184),
    normalize_mean=[0.8047, 0.7808, 0.7769],
    normalize_std=[0.2957, 0.3077, 0.3081],
    device=DEVICE,
)

original, mask = giq.segmentation.extract(
    model=segmenter,
    image_path=str(IMAGE_PATH),
    resize_dim=(1024, 1024),
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    device=DEVICE,
)

coords, confidence, detection = giq.landmark.detect(
    class_name=label,
    class_dict=garment_classes,
    image_path=str(IMAGE_PATH),
    model=landmark_model,
    scale_std=200.0,
    resize_dim=[288, 384],
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    device=DEVICE,
)

refined_coords, refined_detection = giq.landmark.refine(
    class_name=label,
    detection_np=coords,
    detection_conf=confidence,
    detection_dict=detection,
    mask=mask,
)

derived_coords, final_detection = giq.landmark.derive(
    class_name=label,
    detection_dict=refined_detection,
    derivation_dict=derivation_dict,
    landmark_coords=refined_coords,
    np_mask=mask,
)

pixel_measurements, final_detection = giq.utils.compute_measurement_distances(
    final_detection
)
```

For multiple images, prefer the `giq.tailor(...)` class. It centralizes model loading, loops over the input directory, and saves the output artifacts.

---

## 17. Recommended development workflow

### Step 1: Establish a reliable baseline

Start with one garment type and one clear image. Run the individual stages in `full_pipeline.ipynb` and inspect every intermediate result.

### Step 2: Validate classification

Confirm that the predicted label is correct before using the corresponding garment schema. A wrong label causes the wrong landmark indexes and measurement instructions to be selected.

### Step 3: Validate the mask

Plot the segmentation mask. A poor mask produces poor refinement and derivation, even if landmark detection is good.

### Step 4: Validate landmarks visually

Plot original, refined, and derived landmarks over the garment image. Do not trust a new measurement until the points are visually correct.

### Step 5: Add configuration-only measurements first

Try adding a measurement between existing landmarks before changing model code. This is faster, easier to test, and less likely to break other garment classes.

### Step 6: Add derived points when needed

Use the mask and geometry for points that can be defined mathematically. Keep derivation rules garment-specific when necessary.

### Step 7: Retrain only when necessary

Retrain classification if new garment categories are needed. Retrain landmark detection only when a new point cannot be reliably derived.

### Step 8: Add calibration and quality control

For a production system, add:

- real-world scale conversion
- image and mask quality checks
- classifier confidence thresholds
- landmark confidence thresholds
- outlier detection
- manual review for uncertain images
- model and configuration version tracking

### Step 9: Batch with `tailor`

Once the individual stages are stable, use `tailor.py` to process directories consistently and save standard outputs.

---

## 18. Suggested production architecture

```text
Input service
    - upload image
    - validate file and dimensions
    - correct orientation

Classification service
    - predict garment class
    - reject low-confidence results

Segmentation service
    - create garment mask
    - calculate mask quality

Landmark service
    - detect predefined points
    - refine points with the mask
    - derive custom points

Measurement service
    - calculate pixel distances
    - apply calibration
    - calculate uncertainty if available

Validation service
    - check point positions
    - check measurements against expected ranges
    - send uncertain cases for review

Output service
    - measurement JSON
    - annotated image
    - segmentation mask
    - metadata CSV or database record
```

Keep the stages loosely coupled. This lets you replace a classifier, segmentation model, calibration method, or derivation rule without rewriting the complete application.

---

## 19. Common mistakes to avoid

- Do not reuse a classification probability result after changing the image.
- Do not continue with the trousers schema unless the current image was classified as trousers, or the class was deliberately specified.
- Do not assume a good landmark result is possible with a bad mask.
- Do not treat pixel distances as centimeters without calibration.
- Do not change HRNet landmark indexes casually; they are shared model-output positions.
- Do not add a new measurement before confirming that its start and end landmark IDs exist.
- Do not trust a derived point without plotting it on multiple images.
- Do not edit generated output files as a substitute for changing the source pipeline.
- Do not load models repeatedly inside an image loop; load them once and reuse them.

---

## 20. Best starting points

For normal use:

1. Run `full_pipeline.ipynb`.
2. Read `test/tutorial_tailor.ipynb`.
3. Read `test/tutorial_landmark_detection.ipynb`.
4. Read `test/tutorial_landmark_refinement_and_derivation.ipynb`.
5. Read `test/adv_usage_custom_measurement_instruction.ipynb` before adding measurements.

For a custom measurement, the safest order is:

```text
existing landmarks
    -> new instruction measurement
    -> derived landmark if necessary
    -> retrained landmark model only as a last resort
```





Ready for review
Select text to add comments on the plan
src/garmentiq/calibration/ — ArUco pixel → mm metrology
Context
The AI pipeline (classification → segmentation → landmarks) produces pixel coordinates. Nothing converts them to millimetres, so there is no QC verdict possible against a ±3 mm tolerance. A single mm-per-pixel ratio is not good enough: perspective makes a 50 mm marker span a different number of pixels at the centre of the frame than at the corner, and a phone lens bends straight lines exactly where a 1200 mm garment reaches.

This adds a metrology layer that runs beside the AI pipeline and does not change it. The garment lies on a printed ArUco board; every capture detects whatever markers are not occluded, fits a robust planar homography from their corners to their known millimetre positions, and transforms POM pixel points through it. The AI pipeline keeps producing pixels; calibration turns them into millimetres.

src/garmentiq/calibration/ exists but is empty. Written fresh from the pipeline diagram, one concern per file so each can be tested and debugged on its own.

Mode: board in every shot. Each frame computes its own homography, so the camera or table may move between captures. Camera intrinsics are the only thing calibrated once and saved.

Not in scope: wiring this into realtime_measure.py or tailor.py. That is the next step, once the numbers check out.

Architecture
frame ──► undistort.py ──► detect.py ──► homography.py ──► surface.py ──► metric.py ──► mm
              ▲                │              ▲                 ▲
         camera.py        board.py       board.py          camera.py
       (saved once)   (board_config.py)                  (surface offset)
Files
Each module is importable and testable alone. All follow the package conventions: # garmentiq/calibration/<file>.py first line, gerund one-line module docstring, Google-style Args:/Raises:/Returns:, plain ValueError/FileNotFoundError with f-strings, cv2 + numpy only.

File	Responsibility
board.py	BoardSpec dataclass: the physical layout. Builds every marker's 4 corner positions in mm from a regular grid (rows, cols, marker_size_mm, pitch_x_mm, pitch_y_mm, ids row-by-row), or from an explicit {id: (x_mm, y_mm)} map for an irregular board. corners_mm(id) returns (4,2) in OpenCV order TL,TR,BR,BL. to_dict/from_dict/save_json/load_json.
board_config.py	The FILL_THE_DATA file. Every value a FILL_THE_DATA sentinel with a comment saying exactly how to measure it. load_board() returns a BoardSpec and raises a clear error naming any field still unfilled.
detect.py	create_detector(dictionary) with CORNER_REFINE_SUBPIX (sub-pixel accuracy is what buys the ±3 mm), detect_markers(image, ...) -> {id: (4,2) corners}. Rejects duplicate IDs (two boards in view).
camera.py	CameraIntrinsics dataclass (camera_matrix, dist_coeffs, image_size, rms_px). calibrate_camera(detections, board) via cv2.calibrateCameraExtended, dropping views that are the wrong resolution, show too few markers, or reproject worst above ~3× the median. Reports per-view RMS and frame-coverage so a bad capture set is obvious. save_json/load_json.
undistort.py	undistort_image(image, intrinsics) and undistort_points(points_px, intrinsics), both keeping the same pixel frame (same K) so the homography stays valid. Identity pass-through when intrinsics are None — the pipeline runs without lens calibration, just less accurately.
homography.py	estimate_homography(corners_px, board) — stacks every corner of every detected marker into correspondences and fits with cv2.findHomography(..., RANSAC); direct solve when exactly 4 points. Guards: fewer than 2 markers, near-collinear points, degenerate/non-finite H. homography_residuals(...) returns per-corner mm error, rmse_mm, max_mm — the primary health check.
surface.py	lift_to_surface(H, intrinsics, offset_mm). The board plane is the table; landmarks sit on top of the fabric. At ~1170 mm camera height a 5 mm lift is a ~0.43 % scale error ≈ 4.3 mm over a 1 m leg — larger than the whole tolerance, so this is not optional. Decomposes H with K into the plane pose, rebuilds it at z = offset. No-op without intrinsics.
metric.py	pixel_to_mm(points_px, H) (cv2.perspectiveTransform), distance_mm(a, b), local_mm_per_px(points_px, H) from the local homography Jacobian — reports the scale at each POM and flags points in a badly-conditioned region.
calibrate.py	The façade the rest of the app calls: calibrate_frame(image, board, intrinsics=None, surface_offset_mm=0.0) -> FrameCalibration. Runs undistort → detect → homography → lift, and returns H plus diagnostics (marker_ids, rmse_mm, max_mm, n_markers). FrameCalibration.to_mm(points) and .distance(a, b) are the two calls a caller needs.
plot.py	draw_markers(image, corners, ...) outlines and labels every detected marker (red dot on corner 0 so a rotated print is visible at a glance); draw_mm_grid(image, H, ...) back-projects a 100 mm grid onto the photo — if that grid does not sit square on the board, the calibration is wrong and you can see it.
inspect_board.py	Diagnostic, run before filling the config. python -m garmentiq.calibration.inspect_board <photo>: sweeps every predefined ArUco dictionary and reports which one matches, prints the detected IDs arranged in their visual row/column order, estimates marker_size-relative pitch, and writes an annotated overlay. Prints a ready-to-paste board_config.py block.
capture.py	python -m garmentiq.calibration.capture: opens the phone stream (reuses the LatestFrameReader pattern from realtime_measure.py), live-previews marker detection, s saves a frame to calibration_samples/, and tracks which image regions still lack marker coverage so you know when you have enough tilted views for camera.py.
__init__.py	Path comment, prose docstring, flat from .x import y re-exports, matching segmentation/__init__.py. Plus from . import calibration added to src/garmentiq/init.py (cv2+numpy only, cheap to import eagerly).
The data you must fill in
board_config.py ships with these as FILL_THE_DATA, each with a measuring instruction in a comment. inspect_board.py gives you a first estimate for the starred ones; verify against the print with a caliper — a photo cannot reveal printer scaling, since everything scales together.

Field	How to get it
ARUCO_DICTIONARY *	inspect_board.py sweeps and tells you (DICT_4X4_50, DICT_5X5_100, …).
MARKER_IDS *	The ID grid, row by row, e.g. [[0,1,2],[3,4,5]]. inspect_board.py prints it in visual order.
MARKER_SIZE_MM	Caliper across one printed black square including its black border. The nominal 50 mm is usually off by a few tenths — e.g. 50.1.
PITCH_X_MM, PITCH_Y_MM *	Left edge of one marker to the left edge of the next in the same row / column. (board.py also accepts a clear-gap value and adds the marker size for you.)
ROWS, COLS	Count on the print.
BOARD_WIDTH_MM, BOARD_HEIGHT_MM	Optional; used as a sanity cross-check against rows × pitch.
CAMERA_RESOLUTION	Full-res capture size, e.g. (4606, 3456). Intrinsics are only valid at the resolution they were made at.
CAMERA_TO_BOARD_MM	Tape measure, lens to table (~1170 mm on your rig). Used for a sanity check and to predict achievable accuracy.
SURFACE_OFFSET_MM	Fabric thickness — how far the garment top sits above the board. ~5 mm for denim. Directly scales every measurement (see surface.py).
Your new photo shows a grid rather than the four-corner sheet, and I can't count IDs from it reliably — which is why board.py supports both a regular grid and an explicit ID→position map, and inspect_board.py resolves it from a real photo.

Verification
test/test_calibration.py, pure geometry with synthetic data — no camera, no models, no network, runs in seconds. One test class per module so each is debuggable alone (pytest test/test_calibration.py::TestHomography):

board.py — grid generation reproduces hand-computed corners; explicit-map and grid forms agree; JSON round-trips.
detect.py — render a synthetic board with cv2.aruco.generateImageMarker, warp it by a known perspective, detect, and confirm IDs and corner order.
homography.py — build correspondences through a known H, fit, and require recovery to < 0.01 mm. Then corrupt one marker's corners and confirm RANSAC rejects it and rmse_mm stays small. Assert the guards raise on 1 marker / collinear points.
surface.py — a synthetic camera at 1170 mm: confirm a 5 mm lift changes distances by the predicted 0.43 %, and that the no-intrinsics path is an exact no-op.
metric.py — a pure similarity H gives exact known distances; local_mm_per_px matches the analytic scale.
calibrate.py — end-to-end on a synthetic rendered board: known 300 mm segment recovers to < 0.5 mm.
Then on real hardware, in order:

python -m garmentiq.calibration.inspect_board <board photo> → dictionary, ID grid, overlay. Fill board_config.py, verify the marker size with a caliper.
python -m garmentiq.calibration.capture → 20–40 tilted views; calibrate_camera should report RMS < 1 px and > 70 % coverage.
Photograph the bare board flat, run calibrate_frame, check rmse_mm is well under 1 mm and that draw_mm_grid lays a square grid on the board.
Ground truth: put a steel rule on the board, measure a known 500 mm span through FrameCalibration.distance, and confirm the error is inside ±3 mm at several positions — centre, edge and corner of the frame, which is where perspective and distortion bite hardest.
Dependency risk to fix
pyproject.toml pins opencv-python>=4.11.0.86. Your venv resolved OpenCV 5.0.0, where cv2.aruco moved out of contrib and is available — but that floor still permits a 4.x resolution, where cv2.aruco is contrib-only and this whole subpackage fails at import. I will raise the floor to opencv-python>=5.0.0 and add a hasattr(cv2, "aruco") guard in detect.py that raises an actionable message.