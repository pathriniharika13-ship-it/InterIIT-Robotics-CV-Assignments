
import cv2
import numpy as np
import os


PATTERN = (7, 7)          # interior corners of an 8x8 chessboard
MIN_FRAMES = 12
SQUARE_MM = 7.0           # side of one square on the phone screen, in mm
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)


objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Actual resolution:", w, "x", h)


imgpoints = []            # refined 2D corners from each saved view
count = 0

print("SPACE = capture | q = finish")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN, None)

    display = frame.copy()
    if found:
        cv2.drawChessboardCorners(display, PATTERN, corners, found)

    status = f"Captured {count}/{MIN_FRAMES}  |  board: {'FOUND' if found else 'not found'}"
    color = (0, 255, 0) if found else (0, 0, 255)
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("calibration", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:                         # SPACE
        if found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(refined)
            cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{count:02d}.png"), frame)
            count += 1
            print(f"Saved view {count}")
        else:
            print("Corners not found, frame skipped")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

n = len(imgpoints)
print(f"Total views captured: {n}")
if n < MIN_FRAMES:
    raise SystemExit(f"Need at least {MIN_FRAMES} views, got {n}. Run again.")

objpoints = [objp] * n
image_size = (w, h)       # (width, height)

rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

# Own mean reprojection error, per view and overall
total_sq_err, total_pts = 0.0, 0
for i in range(n):
    projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    pts = imgpoints[i].reshape(-1, 2)
    proj = projected.reshape(-1, 2)
    err = np.linalg.norm(pts - proj)
    total_sq_err += err ** 2
    total_pts += len(projected)
    print(f"  view {i:02d}: {err / np.sqrt(len(projected)):.3f} px RMS")

reprojection_error = float(np.sqrt(total_sq_err / total_pts))

print("\nK =\n", K)
print("D =", D)
print(f"OpenCV RMS error:        {rms:.4f} px")
print(f"Own reprojection error:  {reprojection_error:.4f} px")

np.savez("camera_params.npz", K=K, D=D, reprojection_error=reprojection_error)
print("Saved camera_params.npz")
