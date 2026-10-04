

## Task 1.1: Webcam calibration (chessboard on a phone screen)

Calibrating the laptop webcam using an 8x8 chessboard (7x7 interior corners, 7 mm squares) shown on a phone 

- `calibrate_webcam.py`: captures frames with SPACE, detects corners (`findChessboardCorners`), refines them (`cornerSubPix`), runs `calibrateCamera`, and saves `camera_params.npz`.

**Result:** 12 views at 1280x720, re-projection error **0.28 px**. Intrinsics: fx ≈ 768, fy ≈ 767, cx ≈ 631, cy ≈ 364. `camera_params.npz` contains `K` (3x3), `D` (1x5, `[k1, k2, p1, p2, k3]`) and `reprojection_error`

K is valid at the resolution it was calibrated at (1280x720).

## Task 1.2: 3D AR cube on an ArUco marker

- `make_aruco.py`: It generates marker ID 42 from `DICT_6X6_250` (`aruco_42.png`)
- `aruco_3d_pose.py`: loads `camera_params.npz` which were found earlier detects the marker, estimates its pose (R, t) with `solvePnP`, draws a wireframe cube sitting on the marker, and shows the Z distance in cm on the live video.

<img width="385" height="353" alt="image" src="https://github.com/user-attachments/assets/64ec0a15-feb6-4d86-b295-25ccb5f6945e" />



## Task 2: Lane detection with CLAHE

`task2_lane_tracker.py` takes an image path from the terminal:

python task2_lane_tracker.py --path img.png

# The output image

<img width="746" height="556" alt="lane_output" src="https://github.com/user-attachments/assets/1f202bcc-e5a3-44e6-be0d-580ad706a8c4" />

The image shows the output of the given image with the lanes detected which are shown in colour green and yellow . 

## Task 3: ML primitives in PyTorch


- **3.1:** applies a Sobel X kernel with `F.conv2d` to `img.png` and saves the feature map. Relationship between size of output image , kernel size , stride and input size   is floor((in + 2*pad - kernel) / stride) + 1`.
- **3.2:** a `Conv2d(3, 16, 3x3)`, `ReLU` and `MaxPool2d(2, 2)` on a `(1, 3, 416, 416)` tensor.

Printed shapes: `(1, 3, 416, 416)` → Conv `(1, 16, 414, 414)` → ReLU `(1, 16, 414, 414)` → MaxPool `(1, 16, 207, 207)`.

