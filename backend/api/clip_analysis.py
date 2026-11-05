"""
Clip analysis: sync full analysis and async streaming events (shared implementation).
"""
import asyncio
import base64
import json
import sys

import cv2
import numpy as np
import torch

from api import state
from api.coaching_text import clip_coaching_prompt, coaching_lines
from api.config import MAX_VIDEO_DURATION_SECONDS
from api.inference import run_stroke_model


def run_analysis_sync(temp_file: str) -> dict:
    """Core analysis pipeline — blocking; run via asyncio.to_thread from routes."""
    cap = cv2.VideoCapture(temp_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frame_count > 0 and fps > 0:
        estimated_duration = total_frame_count / fps
        if estimated_duration > MAX_VIDEO_DURATION_SECONDS:
            cap.release()
            return {
                "error": (
                    f"This video is {int(estimated_duration // 60)} minutes long. "
                    f"IsoCourt currently supports videos up to {MAX_VIDEO_DURATION_SECONDS // 60} minutes. "
                    "For a full-game analysis, please trim to your key plays or upload quarter by quarter."
                ),
                "over_duration_limit": True,
                "validation_failed": True,
            }

    all_frames_rgb = []
    print("Buffering video frames...")
    sys.stdout.flush()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        all_frames_rgb.append(frame_resized)
        if len(all_frames_rgb) % 60 == 0:
            print(f" -> Buffered {len(all_frames_rgb)} frames...")
            sys.stdout.flush()

    total_frames = len(all_frames_rgb)
    print(f"Buffering complete. Total frames: {total_frames}")
    sys.stdout.flush()

    if total_frames == 0:
        return {"error": "Could not read video frames", "validation_failed": True}

    print("Stage 1: Analyzing poses for badminton characteristics...")
    sys.stdout.flush()

    sample_indices = list(range(0, total_frames, max(1, total_frames // 30)))[:30]
    pose_landmarks_list = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, raw_frame = cap.read()
        if not ret:
            pose_landmarks_list.append([])
            continue

        p_results = state.pose_estimator.process_frame(raw_frame)
        landmarks = state.pose_estimator.get_landmarks_as_list(p_results)
        if landmarks:
            pose_landmarks_list.append(landmarks[0])
        else:
            pose_landmarks_list.append([])

    cap.release()

    is_badminton_pose, pose_confidence, pose_details = state.badminton_detector.is_badminton_video(
        pose_landmarks_list, threshold=0.05
    )

    print(f"Pose Analysis: is_badminton={is_badminton_pose}, confidence={pose_confidence:.3f}")
    sys.stdout.flush()

    if not is_badminton_pose and pose_confidence < 0.05:
        return {
            "error": "This doesn't appear to be a badminton video. Please upload a video showing badminton gameplay with visible players and racket movements.",
            "validation_failed": True,
            "validation_details": {
                "stage": "pose_analysis",
                "pose_confidence": pose_confidence,
                "overhead_score": pose_details.get("overhead_score", 0.0),
                "racket_score": pose_details.get("racket_score", 0.0),
                "stance_score": pose_details.get("stance_score", 0.0),
            },
        }

    timeline = []

    video_duration = total_frames / fps
    if video_duration <= 15:
        window_size_frames = max(int(fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 30:
        window_size_frames = max(int(video_duration / 15 * fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 300:
        target_window = max(video_duration / 6, 5)
        window_size_frames = int(target_window * fps)
        step_size_frames = max(window_size_frames // 2, 1)
    else:
        window_size_frames = int(60 * fps)
        step_size_frames = int(30 * fps)

    print(f"Starting sliding window analysis (FPS: {fps:.2f})...")
    sys.stdout.flush()

    for start in range(0, total_frames - window_size_frames // 2, step_size_frames):
        end = min(start + window_size_frames, total_frames)
        timestamp = f"{int(start / fps) // 60:02d}:{int(start / fps) % 60:02d}"

        indices = np.linspace(start, end - 1, 16).astype(int)
        segment_frames = [all_frames_rgb[idx] for idx in indices]
        segment_tensor = torch.from_numpy(np.array(segment_frames)).float() / 255.0
        segment_tensor = segment_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(state.device)

        middle_idx = indices[len(indices) // 2]

        ext_cap = cv2.VideoCapture(temp_file)
        ext_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, frame_for_pose = ext_cap.read()
        ext_cap.release()

        p_results = None
        annotated_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if ret:
            p_results = state.pose_estimator.process_frame(frame_for_pose)
            annotated_frame = state.pose_estimator.draw_landmarks(frame_for_pose, p_results)

        h, w = annotated_frame.shape[:2]
        scale = 320 / max(h, w)
        pose_img = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode(".jpg", pose_img)
        pose_b64 = base64.b64encode(buffer).decode("utf-8")

        with torch.no_grad():
            outputs = run_stroke_model(segment_tensor, segment_frames)

            seg_results = {}
            for task, logits in outputs.items():
                probs = torch.softmax(logits, dim=1)
                idx_pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, idx_pred].item()

                if task == "quality":
                    quality_map_to_10 = {0: 1, 1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 10}
                    seg_results["quality_numeric"] = quality_map_to_10.get(idx_pred, 5)
                    q_map = {
                        1: "Beginner", 2: "Beginner+", 3: "Developing",
                        4: "Competent", 5: "Competent+", 6: "Proficient",
                        7: "Advanced", 8: "Advanced+", 9: "Expert", 10: "Elite",
                    }
                    seg_results["quality_label"] = q_map.get(seg_results["quality_numeric"], "Unknown")
                elif task == "is_badminton":
                    seg_results["is_badminton_conf"] = conf
                    seg_results["is_badminton_pred"] = idx_pred
                else:
                    seg_results[task] = {
                        "label": state.dataset_metadata[task][idx_pred],
                        "confidence": conf,
                    }

            timeline.append({
                "timestamp": timestamp,
                "label": seg_results["stroke_type"]["label"],
                "confidence": seg_results["stroke_type"]["confidence"],
                "pose_image": pose_b64,
                "is_badminton_conf": seg_results.get("is_badminton_conf", 1.0),
                "is_badminton_pred": seg_results.get("is_badminton_pred", 1),
                "metrics": {
                    "subtype": seg_results.get("stroke_subtype", {"label": "None", "confidence": 0.0}),
                    "technique": seg_results.get("technique", {"label": "Unknown", "confidence": 0.0}),
                    "placement": seg_results.get("placement", {"label": "Unknown", "confidence": 0.0}),
                    "position": seg_results.get("position", {"label": "Unknown", "confidence": 0.0}),
                    "intent": seg_results.get("intent", {"label": "None", "confidence": 0.0}),
                    "quality": seg_results.get("quality_label", "Developing"),
                },
            })

    print(f"Analysis complete. Found {len(timeline)} segments.")
    sys.stdout.flush()

    print("Stage 2: Validating badminton classification from model...")
    sys.stdout.flush()

    badminton_confidences = [seg.get("is_badminton_conf", 0.0) for seg in timeline]
    badminton_predictions = [seg.get("is_badminton_pred", 0) for seg in timeline]

    avg_badminton_conf = np.mean(badminton_confidences) if badminton_confidences else 0.0
    badminton_ratio = sum(badminton_predictions) / len(badminton_predictions) if badminton_predictions else 0.0

    pose_passed = pose_confidence >= 0.30
    model_passed = badminton_ratio >= 0.5 and avg_badminton_conf >= 0.50

    if not (pose_passed or model_passed):
        if pose_confidence < 0.20 and badminton_ratio < 0.3:
            error_msg = "This video doesn't show badminton gameplay. Please upload a video with visible badminton players, racket movements, and court action."
        elif pose_confidence < 0.30:
            error_msg = "Unable to detect badminton-specific movements (overhead strokes, lunges, racket swings). Please ensure the video clearly shows players in action."
        else:
            error_msg = "The video content doesn't match badminton gameplay patterns. Please upload a clear badminton match or practice video."

        return {
            "error": error_msg,
            "validation_failed": True,
            "validation_details": {
                "stage": "model_classification",
                "pose_confidence": pose_confidence,
                "model_confidence": avg_badminton_conf,
                "badminton_ratio": badminton_ratio,
                "pose_passed": pose_passed,
                "model_passed": model_passed,
            },
        }

    print("Validation passed! Proceeding with full analysis...")
    sys.stdout.flush()

    best_event = None
    valid_events = [t for t in timeline if t["label"] != "Other"]
    if valid_events:
        best_event = max(valid_events, key=lambda x: x["confidence"])
    elif timeline:
        best_event = max(timeline, key=lambda x: x["confidence"])

    if best_event:
        action_label = best_event["label"]
        confidence = best_event["confidence"]
        metrics = best_event["metrics"]
        quality_label = metrics["quality"]
        q_rev = {
            "Beginner": 1, "Beginner+": 2, "Developing": 3,
            "Competent": 4, "Competent+": 5, "Proficient": 6,
            "Advanced": 7, "Advanced+": 8, "Expert": 9, "Elite": 10,
        }
        numeric_quality = q_rev.get(quality_label, 5)

        pose_rating = pose_confidence * 10
        weighted_quality = (numeric_quality * 0.9) + (pose_rating * 0.1)
        numeric_quality = int(round(weighted_quality))

        if confidence < 0.3:
            numeric_quality = min(numeric_quality, 4)
            if "(Poor Form/Detection)" not in quality_label:
                quality_label = f"{quality_label} (Low Confidence)"
        elif confidence < 0.5:
            numeric_quality = min(numeric_quality, 7)
    else:
        action_label, confidence, quality_label, numeric_quality = "Other", 0.0, "Developing", 3
        metrics = {k: {"label": "Unknown", "confidence": 0.0} for k in ["subtype", "technique", "placement", "position", "intent"]}
        metrics["quality"] = "Developing"

    reco_subtype = metrics.get("subtype", {}).get("label", "None") if isinstance(metrics.get("subtype"), dict) else "None"
    reco_tech = metrics.get("technique", {}).get("label", "Unknown") if isinstance(metrics.get("technique"), dict) else "Unknown"
    reco_place = metrics.get("placement", {}).get("label", "Unknown") if isinstance(metrics.get("placement"), dict) else "Unknown"
    reco_pos = metrics.get("position", {}).get("label", "Unknown") if isinstance(metrics.get("position"), dict) else "Unknown"
    reco_intent = metrics.get("intent", {}).get("label", "None") if isinstance(metrics.get("intent"), dict) else "None"

    recommendations = []
    if state.gemini_enabled and state.gemini_client:
        try:
            prompt = clip_coaching_prompt(
                action_label, reco_subtype, confidence, metrics,
                quality_label, numeric_quality, timeline,
            )
            response = state.gemini_client.models.generate_content(model=state.gemini_model_name, contents=prompt)
            recommendations = coaching_lines(response.text, 3)
        except Exception:
            recommendations = ["Keep your eye on the shuttle.", "Maintain a low center of gravity.", "Prepare your racket early."]
    else:
        recommendations = ["Focus on early preparation.", "Maintain a balanced ready position."]

    return {
        "action": action_label,
        "confidence": confidence,
        "subtype": reco_subtype,
        "quality": quality_label,
        "quality_numeric": numeric_quality,
        "recommendations": recommendations,
        "tactical_analysis": {
            "technique": metrics.get("technique", {}),
            "placement": metrics.get("placement", {}),
            "position": metrics.get("position", {}),
            "intent": metrics.get("intent", {}),
        },
        "timeline": timeline,
    }


_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_HIP = 23
_RIGHT_HIP = 24


def _largest_box(mask, width, height, min_fraction):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < width * height * min_fraction:
        return None
    return cv2.boundingRect(contour)


def _court_floor_box(frame):
    """Playing surface in pixels: a green court, or a wood gym floor."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    green = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([95, 255, 255]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    box = _largest_box(green, width, height, 0.08)
    if box is not None:
        return box
    wood = cv2.inRange(hsv, np.array([8, 15, 70]), np.array([36, 100, 255]))
    wood[: int(height * 0.32), :] = 0
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return _largest_box(wood, width, height, 0.08)


def _horizontal_court_lines(frame):
    """Horizontal lines as (y, length), longest line kept when several sit together."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    height, width = frame.shape[:2]
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60,
        minLineLength=int(width * 0.28), maxLineGap=18,
    )
    if raw is None:
        return []
    # OpenCV 4 returns (N, 1, 4). OpenCV 5 returns (N, 4).
    rows = []
    for x1, y1, x2, y2 in np.asarray(raw).reshape(-1, 4):
        if abs(int(y2) - int(y1)) > 8:
            continue
        length = abs(int(x2) - int(x1))
        if length < width * 0.28:
            continue
        rows.append((int(round((int(y1) + int(y2)) / 2)), length))
    rows.sort()
    clustered = []
    for y, length in rows:
        if clustered and abs(y - clustered[-1][0]) <= 12:
            if length > clustered[-1][1]:
                clustered[-1] = (y, length)
        else:
            clustered.append((y, length))
    return clustered


def _row_is_floor(frame, y):
    """True when the pixels just below a line are wood or a green court, not wall or ceiling."""
    height, width = frame.shape[:2]
    sample_y = int(min(height - 2, max(0, y + 8)))
    x1, x2 = int(width * 0.15), int(width * 0.85)
    patch = frame[sample_y:sample_y + 2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = float(np.median(hsv[:, :, 0]))
    sat = float(np.median(hsv[:, :, 1]))
    val = float(np.median(hsv[:, :, 2]))
    wood = 8 <= hue <= 40 and 12 <= sat <= 140 and val >= 70
    green = 32 <= hue <= 95 and sat >= 40 and val >= 40
    return wood or green


def _court_reference(frame, foot_y):
    """
    Net, short-service line, and near baseline for a camera behind the player.
    The service line is the floor line just in front of the feet.
    """
    height, width = frame.shape[:2]
    if foot_y < height * 0.55:
        return None
    lines = _horizontal_court_lines(frame)
    floor_lines = [
        y for y, length in lines
        if height * 0.45 < y < height * 0.96 and length >= width * 0.40 and _row_is_floor(frame, y)
    ]
    ahead = [y for y in floor_lines if foot_y - height * 0.14 < y < foot_y - 4]
    if not ahead:
        return None
    service_y = max(ahead)
    behind = [y for y in floor_lines if y > foot_y + 8]
    base_y = max(behind) if behind else int(height * 0.94)
    if base_y <= service_y + 20:
        return None
    # The front court is farther from the camera, so the same metres take fewer pixels.
    front_px = max(28.0, (base_y - service_y) * (1.98 / 4.72) * 0.62)
    net_y = service_y - front_px
    return (float(net_y), float(service_y), float(base_y))


def _depth_on_near_court(foot_y, reference):
    """0 at the net, the short service line opens the front court, 1 at the near baseline."""
    net_y, service_y, base_y = reference
    if foot_y <= service_y:
        span = (service_y - foot_y) / max(1.0, service_y - net_y)
        return float(min(0.30, max(0.05, 0.22 - 0.17 * span)))
    span = (foot_y - service_y) / max(1.0, base_y - service_y)
    return float(min(1.0, max(0.36, 0.36 + 0.64 * span)))


def _center_line_x(frame):
    """Image x of the painted center line on the near court, when one is visible."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    y1 = int(height * 0.62)
    green = (
        (hsv[y1:, :, 0] >= 35) & (hsv[y1:, :, 0] <= 95)
        & (hsv[y1:, :, 1] >= 40) & (hsv[y1:, :, 2] >= 50)
    )
    counts = green.sum(axis=0)
    xs = np.where(counts >= 6)[0]
    if len(xs) < 4:
        return None
    median = float(np.median(xs))
    cluster = xs[np.abs(xs - median) < width * 0.04]
    if len(cluster) < 4:
        return None
    return float(np.median(cluster))


def _joint_px(landmarks, index, width, height):
    by_id = {lm["id"]: lm for lm in landmarks}
    lm = by_id.get(index)
    if not lm:
        return None
    visibility = lm.get("visibility")
    if isinstance(visibility, (int, float)) and visibility < 0.5:
        return None
    return lm["x"] * width, lm["y"] * height


def _foot_on_court(landmarks, frame, box):
    """Feet on the near half. x is left to right. y is 0 at the net and 1 at the baseline."""
    if not landmarks or box is None:
        return None
    height, width = frame.shape[:2]
    ankles = [pt for pt in (
        _joint_px(landmarks, _LEFT_ANKLE, width, height),
        _joint_px(landmarks, _RIGHT_ANKLE, width, height),
    ) if pt]
    if len(ankles) < 1:
        return None
    px = sum(pt[0] for pt in ankles) / len(ankles)
    # The lead foot is the one closer to the net. A lunge has to count.
    py = min(pt[1] for pt in ankles)
    x, y, w, h = box
    if w < 10 or h < 10:
        return None
    cx = (px - x) / w
    if cx < -0.05 or cx > 1.05:
        return None
    return {
        "x": float(min(1.0, max(0.0, cx))),
        "ix": float(px / width),
        "py": float(py),
        "box_y": float((py - y) / h),
    }


def _place_rally_feet(points, references, centers):
    """Map per-frame foot samples onto the court. Inputs are small scalars, not images."""
    reference = None
    if references:
        reference = tuple(
            sorted(item[axis] for item in references)[len(references) // 2]
            for axis in range(3)
        )
    center_frac = None
    if centers:
        width = centers[len(centers) // 2][1]
        center_px = sorted(item[0] for item in centers)[len(centers) // 2]
        center_frac = center_px / width
    placed = []
    for foot in points:
        if reference:
            depth = _depth_on_near_court(foot["py"], reference)
        else:
            depth = foot["box_y"]
        if center_frac is not None:
            half = max(center_frac, 1.0 - center_frac)
            lateral = 0.5 + (foot["ix"] - center_frac) / (2.0 * half)
        else:
            lateral = foot["x"]
        if depth < -0.05 or depth > 1.05 or lateral < -0.05 or lateral > 1.05:
            continue
        placed.append({
            "x": float(min(1.0, max(0.0, lateral))),
            "y": float(min(1.0, max(0.0, depth))),
            "frame": foot["frame"],
        })
    kept = []
    for index, foot in enumerate(placed):
        if 0 < index < len(placed) - 1:
            before = placed[index - 1]["y"]
            after = placed[index + 1]["y"]
            if abs(foot["y"] - before) > 0.28 and abs(foot["y"] - after) > 0.28:
                continue
        kept.append(foot)
    return kept


def _observe_rally_frame(frame, idx):
    """Score one frame. Returns (foot or None, reference or None, center or None, court_seen)."""
    box = _court_floor_box(frame)
    if box is None:
        return None, None, None, False
    detected = state.pose_estimator.process_frame(frame)
    people = state.pose_estimator.get_landmarks_as_list(detected)
    if not people:
        return None, None, None, True
    foot = _foot_on_court(people[0], frame, box)
    if not foot:
        return None, None, None, True
    reference = _court_reference(frame, foot["py"])
    center = _center_line_x(frame)
    center_item = (center, frame.shape[1]) if center is not None else None
    foot["frame"] = int(idx)
    return foot, reference, center_item, True


def sample_rally_feet(temp_file: str):
    """Foot position on the court floor across the rally, about 8 samples a second.

    Each sampled frame is scored and then released. Only foot positions and a few
    court scalars are kept, so memory stays flat as the clip gets longer.
    """
    if state.pose_estimator is None:
        return []
    cap = cv2.VideoCapture(temp_file)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        if total <= 0:
            return []
        step = max(1, int(round(fps / 8)))
        points = []
        references = []
        centers = []
        saw_court = False
        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                foot, reference, center_item, court_seen = _observe_rally_frame(frame, idx)
            finally:
                del frame
            if court_seen:
                saw_court = True
            if foot is None:
                continue
            if reference is not None:
                references.append(reference)
            if center_item is not None:
                centers.append(center_item)
            points.append(foot)
        if not saw_court:
            return []
        return _place_rally_feet(points, references, centers)
    finally:
        cap.release()


def _stream_load_and_pose_stage_sync(temp_file: str) -> dict:
    """
    Blocking OpenCV + pose + badminton gate. Runs in a thread pool so the asyncio
    loop stays responsive for SSE and other workers.
    """
    cap = cv2.VideoCapture(temp_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frame_count > 0 and fps > 0:
        estimated_duration = total_frame_count / fps
        if estimated_duration > MAX_VIDEO_DURATION_SECONDS:
            cap.release()
            return {
                "event": "error",
                "error": (
                    f"This video is {int(estimated_duration // 60)} minutes long. "
                    f"IsoCourt currently supports videos up to {MAX_VIDEO_DURATION_SECONDS // 60} minutes."
                ),
                "over_duration_limit": True,
            }

    all_frames_rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames_rgb.append(cv2.resize(frame_rgb, (224, 224)))
    total_frames = len(all_frames_rgb)

    if total_frames == 0:
        cap.release()
        return {"event": "error", "error": "Could not read video frames", "validation_failed": True}

    sample_indices = list(range(0, total_frames, max(1, total_frames // 30)))[:30]
    pose_landmarks_list = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, raw_frame = cap.read()
        if not ret:
            pose_landmarks_list.append([])
            continue
        p_results = state.pose_estimator.process_frame(raw_frame)
        landmarks = state.pose_estimator.get_landmarks_as_list(p_results)
        pose_landmarks_list.append(landmarks[0] if landmarks else [])

    cap.release()

    is_badminton_pose, pose_confidence, _pose_details = state.badminton_detector.is_badminton_video(
        pose_landmarks_list, threshold=0.05
    )

    if not is_badminton_pose and pose_confidence < 0.05:
        return {"event": "error", "error": "This does not appear to be a badminton video.", "validation_failed": True}

    return {
        "all_frames_rgb": all_frames_rgb,
        "fps": fps,
        "total_frames": total_frames,
        "pose_confidence": pose_confidence,
    }


async def run_analyze_stream_async(temp_file: str, video_hash: str):
    """
    Async streaming pipeline — yields event dicts (same shape as legacy SSE payloads).
    Final yield should be consumed as done event by caller.
    """
    pre = await asyncio.to_thread(_stream_load_and_pose_stage_sync, temp_file)
    if "event" in pre and pre["event"] == "error":
        yield pre
        return

    all_frames_rgb = pre["all_frames_rgb"]
    fps = pre["fps"]
    total_frames = pre["total_frames"]
    pose_confidence = pre["pose_confidence"]

    video_duration = total_frames / fps
    if video_duration <= 15:
        window_size_frames = max(int(fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 30:
        window_size_frames = max(int(video_duration / 15 * fps), 1)
        step_size_frames = window_size_frames
    elif video_duration < 300:
        target_window = max(video_duration / 6, 5)
        window_size_frames = int(target_window * fps)
        step_size_frames = max(window_size_frames // 2, 1)
    else:
        window_size_frames = int(60 * fps)
        step_size_frames = int(30 * fps)
    window_size_frames = max(window_size_frames, 1)

    rally_feet = await asyncio.to_thread(sample_rally_feet, temp_file)

    timeline = []
    window_index = 0

    for start in range(0, total_frames - window_size_frames // 2, step_size_frames):
        end = min(start + window_size_frames, total_frames)
        timestamp = f"{int(start / fps) // 60:02d}:{int(start / fps) % 60:02d}"

        indices = np.linspace(start, end - 1, 16).astype(int)
        segment_frames = [all_frames_rgb[i] for i in indices]
        segment_tensor = torch.from_numpy(np.array(segment_frames)).float() / 255.0
        segment_tensor = segment_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(state.device)

        middle_idx = indices[len(indices) // 2]

        ext_cap = cv2.VideoCapture(temp_file)
        ext_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
        ret, frame_for_pose = ext_cap.read()
        ext_cap.release()

        annotated_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        if ret:
            p_results = await asyncio.to_thread(state.pose_estimator.process_frame, frame_for_pose)
            annotated_frame = state.pose_estimator.draw_landmarks(frame_for_pose, p_results)

        h, w = annotated_frame.shape[:2]
        scale = 320 / max(h, w)
        pose_img = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", pose_img)
        pose_b64 = base64.b64encode(buf).decode("utf-8")

        with torch.no_grad():
            outputs = await asyncio.to_thread(run_stroke_model, segment_tensor, segment_frames)
            seg_results = {}
            for task, logits in outputs.items():
                probs = torch.softmax(logits, dim=1)
                idx_pred = torch.argmax(probs, dim=1).item()
                conf = probs[0, idx_pred].item()
                if task == "quality":
                    qmap = {0: 1, 1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 10}
                    seg_results["quality_numeric"] = qmap.get(idx_pred, 5)
                    ql = {1: "Beginner", 2: "Beginner+", 3: "Developing", 4: "Competent", 5: "Competent+", 6: "Proficient", 7: "Advanced", 8: "Advanced+", 9: "Expert", 10: "Elite"}
                    seg_results["quality_label"] = ql.get(seg_results["quality_numeric"], "Unknown")
                elif task == "is_badminton":
                    seg_results["is_badminton_conf"] = conf
                    seg_results["is_badminton_pred"] = idx_pred
                else:
                    seg_results[task] = {"label": state.dataset_metadata[task][idx_pred], "confidence": conf}

        event_data = {
            "event": "progress",
            "window": window_index,
            "timestamp": timestamp,
            "label": seg_results.get("stroke_type", {}).get("label", "Other"),
            "confidence": seg_results.get("stroke_type", {}).get("confidence", 0.0),
            "pose_image": pose_b64,
            "metrics": {
                "subtype": seg_results.get("stroke_subtype", {"label": "None", "confidence": 0.0}),
                "technique": seg_results.get("technique", {"label": "Unknown", "confidence": 0.0}),
                "placement": seg_results.get("placement", {"label": "Unknown", "confidence": 0.0}),
                "position": seg_results.get("position", {"label": "Unknown", "confidence": 0.0}),
                "intent": seg_results.get("intent", {"label": "None", "confidence": 0.0}),
                "quality": seg_results.get("quality_label", "Developing"),
            },
            "feet": [
                {"x": point["x"], "y": point["y"]}
                for point in rally_feet
                if start <= point["frame"] < end
            ],
        }
        timeline.append(event_data)
        yield event_data
        await asyncio.sleep(0)
        window_index += 1

    clean_timeline = [{k: v for k, v in t.items() if k != "event"} for t in timeline]

    best_event = None
    valid_events = [t for t in timeline if t.get("label") != "Other"]
    if valid_events:
        best_event = max(valid_events, key=lambda x: x.get("confidence", 0))
    elif timeline:
        best_event = max(timeline, key=lambda x: x.get("confidence", 0))

    if best_event:
        action_label = best_event.get("label", "Other")
        confidence = best_event.get("confidence", 0.0)
        metrics = best_event.get("metrics", {})
        quality_label = metrics.get("quality", "Developing")
        q_rev = {
            "Beginner": 1, "Beginner+": 2, "Developing": 3,
            "Competent": 4, "Competent+": 5, "Proficient": 6,
            "Advanced": 7, "Advanced+": 8, "Expert": 9, "Elite": 10,
        }
        numeric_quality = q_rev.get(quality_label, 5)

        pose_rating = pose_confidence * 10
        weighted_quality = (numeric_quality * 0.9) + (pose_rating * 0.1)
        numeric_quality = int(round(weighted_quality))

        if confidence < 0.3:
            numeric_quality = min(numeric_quality, 4)
        elif confidence < 0.5:
            numeric_quality = min(numeric_quality, 7)
    else:
        action_label, confidence, quality_label, numeric_quality = "Other", 0.0, "Developing", 3
        metrics = {k: {"label": "Unknown", "confidence": 0.0} for k in ["subtype", "technique", "placement", "position", "intent"]}
        metrics["quality"] = "Developing"

    reco_subtype = metrics.get("subtype", {}).get("label", "None") if isinstance(metrics.get("subtype"), dict) else "None"
    reco_tech = metrics.get("technique", {}).get("label", "Unknown") if isinstance(metrics.get("technique"), dict) else "Unknown"
    reco_place = metrics.get("placement", {}).get("label", "Unknown") if isinstance(metrics.get("placement"), dict) else "Unknown"
    reco_pos = metrics.get("position", {}).get("label", "Unknown") if isinstance(metrics.get("position"), dict) else "Unknown"
    reco_intent = metrics.get("intent", {}).get("label", "None") if isinstance(metrics.get("intent"), dict) else "None"

    recommendations = []
    if state.gemini_enabled and state.gemini_client:
        try:
            prompt = clip_coaching_prompt(
                action_label, reco_subtype, confidence, metrics,
                quality_label, numeric_quality, timeline,
            )
            gemini_resp = await asyncio.to_thread(
                state.gemini_client.models.generate_content,
                model=state.gemini_model_name,
                contents=prompt,
            )
            recommendations = coaching_lines(gemini_resp.text, 3)
        except Exception:
            recommendations = ["Keep your eye on the shuttle.", "Maintain a low center of gravity.", "Prepare your racket early."]
    else:
        recommendations = ["Focus on early preparation.", "Maintain a balanced ready position."]

    summary_result = {
        "action": action_label,
        "confidence": confidence,
        "subtype": reco_subtype,
        "quality": quality_label,
        "quality_numeric": numeric_quality,
        "recommendations": recommendations,
        "tactical_analysis": {
            "technique": metrics.get("technique", {}),
            "placement": metrics.get("placement", {}),
            "position": metrics.get("position", {}),
            "intent": metrics.get("intent", {}),
        },
        "timeline": clean_timeline,
        "cache_hit": False,
    }
    state._result_cache[video_hash] = summary_result
    yield {"event": "done", "summary": summary_result}


def sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
