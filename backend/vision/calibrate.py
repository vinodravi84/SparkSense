# FILE: calibrate.py

import cv2
import json
import numpy as np
import argparse
import glob
import os
import sys

# ─── Argument Parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Calibration: Manually enter 4 corner coordinates of a known rectangle."
)
parser.add_argument(
    "--video", "-v",
    help="Path to video file. If omitted, uses the most recent file in temp_videos/",
    default=None
)
args = parser.parse_args()

# ─── Determine Video Path ────────────────────────────────────────────────────
if args.video:
    video_path = args.video
else:
    candidates = glob.glob(os.path.join("../temp_videos", "*"))
    if not candidates:
        print("No files found in temp_videos/. Please provide --video <path>.")
        sys.exit(1)
    video_path = max(candidates, key=os.path.getctime)

print(f"[INFO] Using video for calibration: {video_path}")

# ─── Capture One Frame ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()
if not ret:
    print(f"ERROR: Could not read from {video_path}")
    sys.exit(1)

# ─── Save Frame for Manual Annotation ─────────────────────────────────────────
frame_file = "calibration_frame.jpg"
cv2.imwrite(frame_file, frame)
print(f"[INFO] Saved first frame as '{frame_file}'.")
print("  → Open it in an image viewer and note the (x,y) of the 4 corners of a known rectangle, in clockwise order.")

# ─── Prompt for 4 Points ─────────────────────────────────────────────────────
pts = []
for i in range(1, 5):
    coords = input(f"Enter corner {i} as 'x,y': ")
    try:
        x_str, y_str = coords.split(",")
        x, y = int(x_str.strip()), int(y_str.strip())
        pts.append([x, y])
    except:
        print("Invalid format. Please use 'x,y' (e.g. 100,200).")
        sys.exit(1)

# ─── Real-World Dimensions ───────────────────────────────────────────────────
try:
    W = float(input("Enter real-world width of that rectangle (meters): "))
    H = float(input("Enter real-world height of that rectangle (meters): "))
except ValueError:
    print("Invalid number entered. Exiting.")
    sys.exit(1)

# ─── Compute Pixel↔Meter Scale and Homography ─────────────────────────────────
pts_np = np.array(pts, dtype=np.float32)
p0, p1 = pts_np[0], pts_np[1]
pixel_w = np.linalg.norm(p1 - p0)
px_per_m = pixel_w / W

dst = np.array([
    [0, 0],
    [W * px_per_m, 0],
    [W * px_per_m, H * px_per_m],
    [0, H * px_per_m]
], dtype=np.float32)

Hmat, _ = cv2.findHomography(pts_np, dst)

# ─── Save Configuration ───────────────────────────────────────────────────────
cfg = {
    "homography": Hmat.tolist(),
    "px_per_m": px_per_m,
    "warped_size": [int(W * px_per_m), int(H * px_per_m)]
}
with open("config.json", "w") as f:
    json.dump(cfg, f, indent=2)

print(f"[INFO] Calibration complete. Saved to 'config.json'.")
print(f"  → px_per_m = {px_per_m:.2f}")
