import cv2
import numpy as np

MARKER_MM = 46.0          # side of the black frame on the phone screen, in mm
MARKER_ID = 42
DICT = cv2.aruco.DICT_6X6_250

params = np.load("camera_params.npz")
K, D = params["K"], params["D"]

# Marker corners in the marker's own frame (origin at the marker center, Z = 0 plane).
# Order must match the detector: top-left, top-right, bottom-right, bottom-left.
h = MARKER_MM / 2
obj_pts = np.array([[-h,  h, 0],
                    [ h,  h, 0],
                    [ h, -h, 0],
                    [-h, -h, 0]], dtype=np.float32)

CUBE_H = MARKER_MM
cube_pts = np.array([[-h,  h, 0], [ h,  h, 0], [ h, -h, 0], [-h, -h, 0],            # base
                     [-h,  h, CUBE_H], [ h,  h, CUBE_H], [ h, -h, CUBE_H], [-h, -h, CUBE_H]],  # top
                    dtype=np.float32)
base_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
pillar_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]
top_edges = [(4, 5), (5, 6), (6, 7), (7, 4)]

#  ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

# Webcam (same resolution used for calibration) 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Actual resolution:", w, "x", h_img)
if (w, h_img) != (1280, 720):
    raise SystemExit("K was calibrated at 1280x720; resolution must match.")

frame_count = 0
print("q = quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for c, i in zip(corners, ids.flatten()):
            if i != MARKER_ID:
                continue

            img_pts = c.reshape(4, 2).astype(np.float32)
            ok_pnp, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok_pnp:
                continue

            R, _ = cv2.Rodrigues(rvec)            # 3x3 rotation matrix
            z_cm = float(tvec[2][0]) / 10.0       # tvec is in mm -> cm

            # Axes drawn on the marker (length = half the marker side)
            cv2.drawFrameAxes(frame, K, D, rvec, tvec, MARKER_MM / 2)

            cv2.putText(frame, f"Z distance: {z_cm:.1f} cm", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # Project the 8 cube corners into the image and draw the 12 edges
            proj, _ = cv2.projectPoints(cube_pts, rvec, tvec, K, D)
            p = proj.reshape(-1, 2).astype(int)
            for a, b in base_edges:
                cv2.line(frame, tuple(p[a]), tuple(p[b]), (0, 255, 0), 2)
            for a, b in pillar_edges:
                cv2.line(frame, tuple(p[a]), tuple(p[b]), (255, 0, 0), 2)
            for a, b in top_edges:
                cv2.line(frame, tuple(p[a]), tuple(p[b]), (0, 0, 255), 2)

            if frame_count % 10 == 0:
                print(f"Z = {z_cm:.1f} cm | t(mm) = {tvec.ravel().round(1)}")

    cv2.imshow("ArUco pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
