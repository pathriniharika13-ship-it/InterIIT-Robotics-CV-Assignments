import argparse
import cv2
import numpy as np

WORK_W, WORK_H = 746, 556
INNER_TRIM = (0.004, 0.004, 0.005, 0.007)    # left, top, right, bottom (fractions)

CLIP_LIMIT = 3.0          # CLAHE contrast limit
TILE_GRID = (8, 8)        # CLAHE tile grid

# Pass 1: solid lines (wide, bright, long)
SOLID_KERNEL, SOLID_THRESH, SOLID_MIN_LEN = 21, 85, 150
# Pass 2: dashes (thin, can be faint, short)
DASH_KERNEL, DASH_THRESH = 15, 50
DASH_MIN_LEN, DASH_MAX_WIDTH = 15, 14

MIN_AREA = 30
MIN_ASPECT = 3.0
MAX_ANGLE = 50

LOWER_ROAD = [(0.00, 0.52), (0.69, 0.67), (1.00, 0.77), (1.00, 1.00), (0.00, 1.00)]
UPPER_TOP = [(0.00, 0.42), (0.20, 0.44), (0.44, 0.48), (0.63, 0.51), (1.00, 0.56)]
UPPER_BOT = [(0.00, 0.45), (0.66, 0.57), (1.00, 0.61)]
UPPER_X_RANGES = [(0.00, 0.23), (0.33, 1.00)]


def auto_crop(img, white=235, min_frac=0.3):
  
    content = (img < white).any(axis=2)
    rows = np.where(content.mean(axis=1) > min_frac)[0]
    cols = np.where(content.mean(axis=0) > min_frac)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def apply_clahe(bgr):
  
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID).apply(l)
    return l_eq, cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def road_mask(h, w):
    mask = np.zeros((h, w), np.uint8)
    lower = np.array([[int(x * w), int(y * h)] for x, y in LOWER_ROAD], np.int32)
    cv2.fillPoly(mask, [lower], 255)

    top_x, top_y = zip(*UPPER_TOP)
    bot_x, bot_y = zip(*UPPER_BOT)
    for x_start, x_end in UPPER_X_RANGES:
        xs = np.linspace(x_start, x_end, 20)
        top = [[int(x * w), int(np.interp(x, top_x, top_y) * h)] for x in xs]
        bot = [[int(x * w), int(np.interp(x, bot_x, bot_y) * h)] for x in xs[::-1]]
        cv2.fillPoly(mask, [np.array(top + bot, np.int32)], 255)
    return mask


def find_segments(l_eq, mask, kernel, thresh, min_len, max_width=None):
  
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    tophat = cv2.morphologyEx(l_eq, cv2.MORPH_TOPHAT, k)
    _, bw = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)
    bw = cv2.bitwise_and(bw, mask)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    segments = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
        _, (rw, rh), _ = cv2.minAreaRect(pts)
        length, width = max(rw, rh), min(rw, rh)
        if width < 1 or length < min_len or length / width < MIN_ASPECT:
            continue
        if max_width is not None and width > max_width:
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        if abs(np.degrees(np.arctan(vy / vx if vx else 1e9))) > MAX_ANGLE:
            continue
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        p1 = (int(x0 + t.min() * vx), int(y0 + t.min() * vy))
        p2 = (int(x0 + t.max() * vx), int(y0 + t.max() * vy))
        segments.append((p1, p2, length))
    return segments


def extend_line(p1, p2, w, h):
  
    p1 = np.array(p1, np.float64)
    p2 = np.array(p2, np.float64)
    d = (p2 - p1) / np.linalg.norm(p2 - p1)

    def walk(p, direction):
        last = p.copy()
        for s in range(0, 3000):
            q = p + direction * s
            if not (0 <= q[0] < w and 0 <= q[1] < h):
                break
            last = q
        return last

    a = walk(p1, -d)
    b = walk(p2, d)
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def detect_lanes(l_eq, mask):
    solid = find_segments(l_eq, mask, SOLID_KERNEL, SOLID_THRESH, SOLID_MIN_LEN)
    dashes = find_segments(l_eq, mask, DASH_KERNEL, DASH_THRESH, DASH_MIN_LEN, DASH_MAX_WIDTH)

    # Drop dashes that lie on a solid line already found
    solid_img = np.zeros(l_eq.shape, np.uint8)
    for p1, p2, _ in solid:
        cv2.line(solid_img, p1, p2, 255, 15)
    dashes = [d for d in dashes
              if solid_img[(d[0][1] + d[1][1]) // 2, (d[0][0] + d[1][0]) // 2] == 0]

    # Extend each solid line to the image borders (delete this line to disable)
    h, w = l_eq.shape
    solid = [(*extend_line(p1, p2, w, h), length) for p1, p2, length in solid]
    return solid, dashes


def label(panel, text):
    out = panel.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Lane detection using CLAHE")
    parser.add_argument("--path", required=True, help="path to the input image")
    parser.add_argument("--out", default="lane_output.png", help="where to save the result")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="display scale for the 3-panel window (lower = smaller window)")
    args = parser.parse_args()

    img = cv2.imread(args.path)
    if img is None:
        raise SystemExit(f"Could not read image: {args.path}")

    img = auto_crop(img)
    h0, w0 = img.shape[:2]
    l, t, r, b = INNER_TRIM
    img = img[int(t * h0):int((1 - b) * h0), int(l * w0):int((1 - r) * w0)]
    img = cv2.resize(img, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    l_eq, enhanced = apply_clahe(img)
    mask = road_mask(h, w)
    solid, dashes = detect_lanes(l_eq, mask)

    result = img.copy()
    for p1, p2, _ in solid:
        cv2.line(result, p1, p2, (0, 255, 255), 4)
    for p1, p2, _ in dashes:
        cv2.line(result, p1, p2, (0, 255, 0), 4)

    print(f"Detected {len(solid)} solid lines and {len(dashes)} dashes")
    cv2.imwrite(args.out, result)
    print("Saved", args.out)

   
    combo = np.hstack([label(img, "Original"),
                       label(enhanced, "CLAHE"),
                       label(result, "Lanes")])
    cv2.imwrite("comparison.png", combo)
  

   
    shown = cv2.resize(combo, None, fx=args.scale, fy=args.scale,
                       interpolation=cv2.INTER_AREA)
    win = "Original | CLAHE | Lanes (any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, shown.shape[1], shown.shape[0])
    cv2.imshow(win, shown)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
