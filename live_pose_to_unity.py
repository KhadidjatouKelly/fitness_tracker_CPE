#!/usr/bin/env python3
"""
LIVE pose → Unity for 4 exercises (honest, angle-based reps):

Exercises:
  - curl
  - front_raise
  - front_punch
  - melee_swing

Flow:
  - Python starts once: webcam + MediaPipe
  - Unity sends {"exercise": "..."} on UDP port 5006 when user clicks a button
  - Python switches form + rep logic depending on current_exercise
  - Sends UDP JSON to Unity: exercise, posture_label, feedback, rep_count, angles
  - Shows annotated window (if --show)
  - Saves annotated MP4 + JSON summary when you press Q

Run:
    python3 live_pose_to_unity.py --out_dir sessions --show
"""

import argparse
import time
import json
import socket
import threading
from pathlib import Path
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

# ---------------- NETWORK ----------------
UDP_OUT_IP = "172.20.10.7"   # Unity machine (127.0.0.1 if same PC)  172.20.10.7
UDP_OUT_PORT = 5005        # Python -> Unity pose data

UDP_CONTROL_PORT = 5006    # Unity -> Python exercise selection

sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


current_exercise = "curl"
exercise_lock = threading.Lock()
VALID_EXERCISES = ["curl", "front_raise", "front_punch", "melee_swing"]

# ---------------- GEOMETRY ----------------
POSE_MAP = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
}

def angle_3pts(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax - bx, ay - by], float)
    v2 = np.array([cx - bx, cy - by], float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return None
    cosang = float(np.dot(v1, v2) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

def mp_landmarks_to_xy(landmarks, width, height):
    pts = {}
    for name, idx in POSE_MAP.items():
        lm = landmarks[idx]
        x, y, v = lm.x, lm.y, lm.visibility
        if v is not None and v > 0.5 and 0 <= x <= 1 and 0 <= y <= 1:
            pts[name] = (x * width, y * height)
        else:
            pts[name] = None
    return pts

def compute_elbow_angles(pts):
    L = R = None
    if pts["left_shoulder"] and pts["left_elbow"] and pts["left_wrist"]:
        L = angle_3pts(pts["left_shoulder"], pts["left_elbow"], pts["left_wrist"])
    if pts["right_shoulder"] and pts["right_elbow"] and pts["right_wrist"]:
        R = angle_3pts(pts["right_shoulder"], pts["right_elbow"], pts["right_wrist"])
    return L, R

def shoulder_flex_angle(pts, side):
    """
    Angle at shoulder: hip -> shoulder -> wrist
    ~180° arm down at side
    smaller angle = arm raised in front
    """
    sh = pts.get(f"{side}_shoulder")
    hip = pts.get(f"{side}_hip")
    wr  = pts.get(f"{side}_wrist")
    if sh and hip and wr:
        return angle_3pts(hip, sh, wr)
    return None

# ---------------- CLASSIFIERS (for feedback only) ----------------
def classify_curl(pts):
    """
    Form classification for curls (feedback).
    Reps will still be strictly angle-cycle based.
    """
    L, R = compute_elbow_angles(pts)
    vals = [x for x in (L, R) if x is not None]
    if not vals:
        return None, "needs_correction", "Arms not clearly visible."

    ang = min(vals)
    elbow_bottom = 70   # flexed top
    elbow_top    = 160  # extended bottom

    if elbow_bottom <= ang <= elbow_top:
        return ang, "good", "Nice controlled curl."
    if ang < elbow_bottom:
        return ang, "needs_correction", "Control the top (don’t cut range)."
    if ang > elbow_top:
        return ang, "needs_correction", "Extend fully at the bottom (no hyperextension)."
    return ang, "needs_correction", "Adjust elbow range of motion."

def classify_front_raise(pts, shoulder_min):
    if shoulder_min is None:
        return None, "needs_correction", "Arms/shoulders not visible."

    # relaxed but not silly
    if 50 <= shoulder_min <= 120:
        return shoulder_min, "good", "Good height for front raise."
    if shoulder_min < 50:
        return shoulder_min, "needs_correction", "Don’t swing too high."
    return shoulder_min, "needs_correction", "Raise to chest/eye height."

def classify_front_punch(pts, shoulder_min, elbow_min):
    if shoulder_min is None or elbow_min is None:
        return None, "needs_correction", "Upper body not clearly visible."

    elbow_ok    = 150 <= elbow_min <= 180
    shoulder_ok = 40  <= shoulder_min <= 120

    if elbow_ok and shoulder_ok:
        return elbow_min, "good", "Solid forward punch."
    msgs = []
    if not elbow_ok:
        msgs.append("Punch with a mostly straight elbow (no floppy arm).")
    if not shoulder_ok:
        msgs.append("Punch at chest height, not too low/high.")
    return elbow_min, "needs_correction", "; ".join(msgs)

def classify_melee_swing(pts, shoulder_min, elbow_min):
    if shoulder_min is None or elbow_min is None:
        return None, "needs_correction", "Arms not clearly visible."

    elbow_ok    = 90  <= elbow_min <= 160
    shoulder_ok = 60  <= shoulder_min <= 150

    if elbow_ok and shoulder_ok:
        return elbow_min, "good", "Nice controlled swing."
    msgs = []
    if not elbow_ok:
        msgs.append("Keep a comfortable bend in your arm.")
    if not shoulder_ok:
        msgs.append("Swing through a mid-range arc, not too low or above head.")
    return elbow_min, "needs_correction", "; ".join(msgs)

# ---------------- GENERIC ANGLE-BASED REP COUNTER ----------------
class RangeRepCounter:
    """
    Generic HIGH->LOW->HIGH cycle-based rep counter.

    - We track a numeric feature per frame (angle).
    - 'HIGH' ~ resting position (e.g., arm down, elbow extended).
    - 'LOW'  ~ contracted/raised position (e.g., arm up, elbow flexed).

    A rep is only counted when:
        HIGH  -> enough frames below 'low_threshold'  -> switch to LOW
        then  LOW   -> value goes above 'high_threshold' -> back to HIGH + rep++

    This is exactly the pattern you already had for curls, generalized.
    """
    def __init__(self, low_threshold, high_threshold, min_low_frames=3):
        self.low = low_threshold
        self.high = high_threshold
        self.min_low_frames = min_low_frames

        self.state = "HIGH"     # start at rest
        self.low_frames = 0
        self.rep_count = 0

    def update(self, val):
        """
        val = current angle for this frame (float or None).

        Returns True iff a NEW rep was counted on this frame.
        """
        if val is None:
            # Missing data → we don't change state, but also don't count reps.
            return False

        new_rep = False

        if self.state == "HIGH":
            # We are in resting zone (e.g., arm down).
            # Check if we stayed "low" for long enough → move to contracted phase.
            if val < self.low:
                self.low_frames += 1
                if self.low_frames >= self.min_low_frames:
                    self.state = "LOW"
            else:
                self.low_frames = 0
        else:  # state == "LOW"
            # We are in contracted phase; once we go back above high → count rep.
            if val > self.high:
                self.state = "HIGH"
                self.low_frames = 0
                self.rep_count += 1
                new_rep = True

        return new_rep

def make_curl_counter():
    # elbow angle: HIGH ~ extended ~ 160+, LOW ~ flexed ~ < 80
    return RangeRepCounter(low_threshold=80.0, high_threshold=150.0, min_low_frames=3)

def make_front_raise_counter():
    # shoulder angle: arm down ~ 180, top ~ 60–90
    # rep when angle dips below ~95 and then returns above ~150
    return RangeRepCounter(low_threshold=95.0, high_threshold=150.0, min_low_frames=3)

def make_front_punch_counter():
    # shoulder angle: punch out ~ smaller angle, retract ~ larger
    # treat like raise but with slightly different thresholds
    return RangeRepCounter(low_threshold=80.0, high_threshold=140.0, min_low_frames=2)

def make_melee_swing_counter():
    # use elbow angle: swing: bent (< 110) then extend (> 150)
    return RangeRepCounter(low_threshold=110.0, high_threshold=150.0, min_low_frames=2)

# ---------------- DRAWING ----------------
SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]

def draw_skeleton(img, pts):
    for a, b in SKELETON:
        pa, pb = pts.get(a), pts.get(b)
        if pa is not None and pb is not None:
            cv2.line(img, (int(pa[0]), int(pa[1])),
                     (int(pb[0]), int(pb[1])), (0, 255, 0), 2)
    for p in pts.values():
        if p is not None:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)

# ---------------- CONTROL THREAD ----------------
def control_loop():
    global current_exercise
    sock_ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_ctrl.bind(("", UDP_CONTROL_PORT))
    print(f"[CTRL] Listening for exercise selection on UDP port {UDP_CONTROL_PORT}...")
    while True:
        try:
            data, addr = sock_ctrl.recvfrom(1024)
            txt = data.decode("utf-8")
            msg = json.loads(txt)
            ex = msg.get("exercise")
            if ex in VALID_EXERCISES:
                with exercise_lock:
                    current_exercise = ex
                print(f"[CTRL] Exercise switched to '{ex}' from {addr}")
            else:
                print(f"[CTRL] Ignoring unknown exercise '{ex}' from {addr}")
        except Exception as e:
            print("[CTRL] Error:", e)

# ---------------- MAIN LOOP ----------------
def run_live(out_dir: str, show: bool):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam 0")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    video_id = f"session_{ts_str}"
    video_path = outp / f"{video_id}_annotated.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    mp_pose = mp.solutions.pose

    # one counter at a time, but angle-based for each exercise
    with exercise_lock:
        ex = current_exercise
    curl_counter        = make_curl_counter()
    raise_counter       = make_front_raise_counter()
    punch_counter       = make_front_punch_counter()
    melee_counter       = make_melee_swing_counter()
    last_exercise = ex

    total_frames = 0
    good_frames = 0
    bad_frames = 0
    mistake_counter = Counter()
    exercises_seen = set()

    print(f"[LIVE] Session '{video_id}' (dynamic exercises from Unity)")
    print("[LIVE] Press Q in the window to end & save.")

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                t_now = time.time()
                total_frames += 1

                with exercise_lock:
                    ex = current_exercise

                exercises_seen.add(ex)

                frame_flipped = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                pts = {k: None for k in POSE_MAP.keys()}
                if res.pose_landmarks:
                    pts = mp_landmarks_to_xy(res.pose_landmarks.landmark, width, height)

                # Basic angles
                L_el, R_el = compute_elbow_angles(pts)
                L_sh = shoulder_flex_angle(pts, "left")
                R_sh = shoulder_flex_angle(pts, "right")

                elbow_min = None
                shoulder_min = None

                elbows = [x for x in (L_el, R_el) if x is not None]
                if elbows:
                    elbow_min = min(elbows)

                shoulders = [x for x in (L_sh, R_sh) if x is not None]
                if shoulders:
                    shoulder_min = min(shoulders)

                posture_label = "needs_correction"
                feedback = "Move into view of the camera."
                feature_val = None
                rep_count = 0

                # ---- per-exercise logic ----
                if ex == "curl":
                    feature_val, posture_label, feedback = classify_curl(pts)
                    curl_counter.update(feature_val)
                    rep_count = curl_counter.rep_count

                elif ex == "front_raise":
                    feature_val, posture_label, feedback = classify_front_raise(pts, shoulder_min)
                    raise_counter.update(shoulder_min)
                    rep_count = raise_counter.rep_count

                elif ex == "front_punch":
                    feature_val, posture_label, feedback = classify_front_punch(pts, shoulder_min, elbow_min)
                    punch_counter.update(shoulder_min)
                    rep_count = punch_counter.rep_count

                elif ex == "melee_swing":
                    feature_val, posture_label, feedback = classify_melee_swing(pts, shoulder_min, elbow_min)
                    melee_counter.update(elbow_min)
                    rep_count = melee_counter.rep_count

                # ---- stats for summary ----
                if posture_label == "good":
                    good_frames += 1
                else:
                    bad_frames += 1
                    if feedback:
                        mistake_counter[feedback] += 1

                # ---- SEND TO UNITY ----
                msg = {
                    "timestamp": t_now,
                    "exercise": ex,
                    "posture_label": posture_label,
                    "feedback": feedback,
                    "rep_count": rep_count,
                    "angles": {
                        "left_elbow": float(L_el) if L_el is not None else None,
                        "right_elbow": float(R_el) if R_el is not None else None,
                        "left_shoulder": float(L_sh) if L_sh is not None else None,
                        "right_shoulder": float(R_sh) if R_sh is not None else None,
                    }
                }
                sock_out.sendto(json.dumps(msg).encode("utf-8"), (UDP_OUT_IP, UDP_OUT_PORT))

                # ---- DRAW ----
                draw_skeleton(frame_flipped, pts)
                cv2.putText(frame_flipped, f"{ex} | reps={rep_count}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
                cv2.putText(frame_flipped, posture_label,
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (50, 255, 50) if posture_label == "good" else (60, 60, 255), 2)
                cv2.putText(frame_flipped, "Press Q to end session",
                            (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                writer.write(frame_flipped)

                if show:
                    cv2.imshow("Live Pose", frame_flipped)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    total_frames = max(total_frames, 1)
    good_pct = good_frames / total_frames

    common_mistakes = []
    for text, cnt in mistake_counter.most_common(3):
        common_mistakes.append({"feedback": text, "count": int(cnt)})

    summary = {
        "video_id": video_id,
        "exercises_seen": list(exercises_seen),
        "total_frames": int(total_frames),
        "good_frames": int(good_frames),
        "bad_frames": int(bad_frames),
        "good_form_ratio": float(good_pct),
        "common_mistakes": common_mistakes,
        "video_path": str(video_path),
        "ended_at": time.time(),
    }

    summary_path = outp / f"{video_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[LIVE] Saved video:", video_path)
    print("[LIVE] Saved summary:", summary_path)
    print("[LIVE] Good%:", round(summary["good_form_ratio"] * 100, 1))


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="sessions", help="Where to save video + summary")
    ap.add_argument("--show", action="store_true", help="Show annotated window")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    t = threading.Thread(target=control_loop, daemon=True)
    t.start()
    run_live(args.out_dir, args.show)
