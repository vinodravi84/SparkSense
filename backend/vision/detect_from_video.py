# ─── IMPORTS ───
import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque, defaultdict

# ─── CONFIGURATION ───
YOLO_MODEL = "yolov8m.pt"
DETECT_SIZE = 640
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
FPS = 30.0
AVG_CAR_WIDTH_M = 2.0
CALIBRATE_SAMPLES = 75
SMOOTH_WINDOW = 7
SPEED_LIMIT = 50.0
BLINK_INTERVAL = 10
BOX_THICKNESS = 3
TEXT_THICKNESS = 2
MIN_PERSISTENCE = 2
MIN_BOX_AREA = 40 * 40
TRAIL_LENGTH = 30

# ─── MODELS ───
model = YOLO(YOLO_MODEL)
tracker = DeepSort(max_age=20, n_init=5, max_cosine_distance=0.15, nn_budget=100)

def get_color(tid):
    np.random.seed(int(tid) % 10000)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]), min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW, interH = max(0, xB - xA), max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    return interArea / float(boxA[2]*boxA[3] + boxB[2]*boxB[3] - interArea)

def estimate_homography(car_boxes):
    image_pts, world_pts = [], []
    car_index = 0
    for box in car_boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        if w < 20:
            continue
        y = y2
        image_pts += [[x1, y], [x2, y]]
        world_pts += [[0, car_index * 2.5], [AVG_CAR_WIDTH_M, car_index * 2.5]]
        car_index += 1
    if len(image_pts) >= 4:
        H, _ = cv2.findHomography(np.array(image_pts, dtype=np.float32),
                                  np.array(world_pts, dtype=np.float32), 0)
        return H
    return None

def map_to_world(H, point):
    p = np.array([point[0], point[1], 1.0])
    wp = H @ p
    if wp[2] == 0:
        return (0, 0)
    return (wp[0] / wp[2], wp[1] / wp[2])

def calibrate_ppm(video_path):
    cap = cv2.VideoCapture(video_path)
    widths = []
    while len(widths) < CALIBRATE_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break
        res = model(frame, conf=0.3, iou=0.45)[0]
        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            if conf < 0.3 or model.names[int(cls)] != "car":
                continue
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            widths.append(x2 - x1)
    cap.release()
    return np.median(widths) / AVG_CAR_WIDTH_M if widths else 300.0

# ─── MAIN FUNCTION ───
def process_frame_stream(video_path):
    start_time = time.time()
    ppm = calibrate_ppm(video_path)
    H = None

    cap = cv2.VideoCapture(video_path)
    W, H_vid = int(cap.get(3)), int(cap.get(4))
    line_y = int(H_vid * 0.5)

    base, _ = os.path.splitext(video_path)
    writer = cv2.VideoWriter(f"{base}_processed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H_vid))

    world_pos = {}
    world_pos_img = {}
    max_speeds = {}
    box_history = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
    height_history = defaultdict(lambda: deque(maxlen=5))
    trail_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    persistence = defaultdict(int)
    car_boxes_for_H_all = []
    speed_zone_active_ids = set()
    crossed_status = {}
    in_count = 0
    out_count = 0
    counted_ids = set()
    frame_count = 0
    allowed = {"car", "truck", "bus", "motorcycle", "bicycle"}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        inp = cv2.resize(frame, (DETECT_SIZE, DETECT_SIZE))
        res = model.predict(inp, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
        x_scale, y_scale = W / DETECT_SIZE, H_vid / DETECT_SIZE
        dets_raw = []

        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            if conf < CONF_THRESHOLD:
                continue
            label = model.names[int(cls)]
            if label not in allowed:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            bx = int(x1 * x_scale)
            by = int(y1 * y_scale)
            bw = int((x2 - x1) * x_scale)
            bh = int((y2 - y1) * y_scale)
            if bw * bh < MIN_BOX_AREA:
                continue
            dets_raw.append(([bx, by, bw, bh], float(conf), label))
            if label == "car":
                car_boxes_for_H_all.append([bx, by, bx + bw, by + bh])

        if frame_count % 10 == 0 and H is None and len(car_boxes_for_H_all) >= 4:
            H = estimate_homography(car_boxes_for_H_all)
            print(f"[CALIBRATION] Homography {'estimated' if H is not None else 'failed'} at frame {frame_count}")

        dets = []
        for i, (boxA, confA, _) in enumerate(dets_raw):
            keep = True
            for j, (boxB, confB, _) in enumerate(dets_raw):
                if i != j and iou(boxA, boxB) > 0.7 and confA < confB:
                    keep = False
                    break
            if keep:
                dets.append(dets_raw[i])

        cv2.circle(frame, (int(W * 0.05), line_y), 6, (0, 255, 255), -1)
        cv2.circle(frame, (int(W * 0.95), line_y), 6, (0, 255, 255), -1)

        tracks = tracker.update_tracks(dets, frame=frame)
        current_ids = set()

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            cx, cy = (l + r) // 2, b
            current_ids.add(tid)
            persistence[tid] += 1
            box_history[tid].append((l, t, r, b))
            trail_history[tid].append((cx, cy))
            height = b - t
            height_history[tid].append(height)
            sl, st, sr, sb = np.mean(box_history[tid], axis=0).astype(int)
            col = get_color(tid)
            cv2.rectangle(frame, (sl, st), (sr, sb), col, BOX_THICKNESS)

            # IN/OUT count logic
            if tid not in crossed_status:
                crossed_status[tid] = cy
            delta_y = cy - crossed_status[tid]
            crossed_status[tid] = cy
            if abs(delta_y) > 5 and tid not in counted_ids:
                counted_ids.add(tid)
                if delta_y > 0:
                    in_count += 1
                    print(f"[FRAME {frame_count}] ID{tid} → IN (ΔY={delta_y:.1f})")
                else:
                    out_count += 1
                    print(f"[FRAME {frame_count}] ID{tid} → OUT (ΔY={delta_y:.1f})")

            # Speed zone logic
            if cy >= line_y:
                speed_zone_active_ids.add(tid)
            if tid in speed_zone_active_ids:
                wpos = map_to_world(H, (cx, cy)) if H is not None else (cx / ppm, cy / ppm)
                if tid in world_pos:
                    old_img = np.array(world_pos_img.get(tid, (cx, cy)))
                    new_img = np.array([cx, cy])
                    image_dist = np.linalg.norm(new_img - old_img)
                    world_pos_img[tid] = (cx, cy)
                    old = np.array(world_pos[tid])
                    new = np.array(wpos)
                    dist = np.linalg.norm(new - old)
                    if image_dist < 2:
                        dist = 0.0
                    if dist < 0.01 and len(height_history[tid]) >= 3:
                        dh = np.abs(np.diff(list(height_history[tid])))
                        avg_dh = np.mean(dh)
                        avg_h = np.mean(height_history[tid])
                        speed = round(np.clip((avg_dh / avg_h) * 80 * FPS, 0, 55), 1) if avg_h > 0 else 0.0
                        print(f"[DEBUG][F{frame_count}] ID{tid} fallback: ΔH={avg_dh:.1f}, H={avg_h:.1f}, Speed={speed:.1f} km/h")
                    else:
                        speed = round((dist / (1.0 / FPS)) * 3.6, 1)
                        if speed > 90:
                            print(f"[WARN][F{frame_count}] ID{tid} speed too high: {speed} km/h, skipped.")
                            speed = 0.0
                        else:
                            print(f"[DEBUG][F{frame_count}] ID{tid} Δm={dist:.2f}, Speed={speed:.1f} km/h")
                    prev = max_speeds.get(tid, 0)
                    if speed - prev > 20:
                        speed = prev + 5
                    max_speeds[tid] = max(speed, prev)
                else:
                    max_speeds[tid] = 0.0
                    world_pos_img[tid] = (cx, cy)
                world_pos[tid] = wpos
                speed = max_speeds[tid]
                if persistence[tid] >= MIN_PERSISTENCE:
                    color = (0, 0, 255) if speed > SPEED_LIMIT and (frame_count // BLINK_INTERVAL) % 2 == 0 else col
                    cv2.putText(frame, f"ID{tid} {speed:.1f} km/h", (sl, st - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, TEXT_THICKNESS)

            # Draw arrow trail
            for i in range(1, len(trail_history[tid])):
                pt1 = trail_history[tid][i - 1]
                pt2 = trail_history[tid][i]
                alpha = int(255 * (i / len(trail_history[tid])))
                color = tuple(int(c * (i / len(trail_history[tid]))) for c in get_color(tid))
                cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.3)

        for tid in list(persistence):
            if tid not in current_ids:
                persistence[tid] = 0

        cv2.putText(frame, f"IN: {in_count}  OUT: {out_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        writer.write(frame)
        _, jpg = cv2.imencode('.jpg', frame)
        yield jpg.tobytes()

    cap.release()
    writer.release()
    print(f"[STREAM] Output saved to: {base}_processed.mp4")
    print(f"[TIME] Total processing time: {time.time() - start_time:.2f} seconds")
