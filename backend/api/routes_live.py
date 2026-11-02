"""Live session: capacity gate + WebSocket frame-by-frame analysis."""
from __future__ import annotations

import asyncio
import base64
import json
import uuid

import cv2
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api import state
from api.coaching_text import live_coaching_prompt, live_cue_worth_saying, plain_coaching_text
from api.inference import run_stroke_model

router = APIRouter(tags=["live"])


@router.post("/sessions")
async def start_live_session():
    """
    Reserve a live coaching slot. Returns 503 when MAX_CONCURRENT_LIVE_SESSIONS is reached.
    """
    if state._live_semaphore._value <= 0:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "IsoCourt cannot start a live session right now — capacity limits reached.",
                "retry_after": 120,
            },
        )
    await state._live_semaphore.acquire()
    session_id = str(uuid.uuid4())
    async with state._live_lock:
        state._live_sessions[session_id] = None
    return {"session_id": session_id, "status": "ready"}


@router.delete("/sessions/{session_id}")
async def end_live_session(session_id: str):
    async with state._live_lock:
        if session_id in state._live_sessions:
            del state._live_sessions[session_id]
            state._live_semaphore.release()
    return {"ok": True}


def _decode_frame(raw: bytes) -> np.ndarray | None:
    """Decode a JPEG/PNG blob into a BGR numpy array."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


STROKE_CONFIDENCE_FLOOR = 0.25
BADMINTON_PROB_MIN = 0.55
POSE_CONF_MIN = 0.05
RING_SIZE = 16
POSE_SAMPLE_COUNT = 4
POSE_CHECK_INTERVAL = 10
INFERENCE_INTERVAL_S = 1.0


def _run_window_inference(frames_rgb: list[np.ndarray]) -> dict:
    """Run the sliding-window model on exactly 16 (224x224) RGB frames. Blocking."""
    segment = np.array(frames_rgb)  # (16, 224, 224, 3)
    tensor = torch.from_numpy(segment).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).to(state.device)  # (1, 16, 3, 224, 224)

    with torch.no_grad():
        outputs = run_stroke_model(tensor, frames_rgb)

    seg_results: dict = {}
    for task, logits in outputs.items():
        probs = torch.softmax(logits, dim=1)
        idx_pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, idx_pred].item()

        if task == "quality":
            qmap = {0: 1, 1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 10}
            seg_results["quality_numeric"] = qmap.get(idx_pred, 5)
            ql = {1: "Beginner", 2: "Beginner+", 3: "Developing", 4: "Competent",
                  5: "Competent+", 6: "Proficient", 7: "Advanced", 8: "Advanced+",
                  9: "Expert", 10: "Elite"}
            seg_results["quality_label"] = ql.get(seg_results["quality_numeric"], "Unknown")
        elif task == "is_badminton":
            seg_results["badminton_prob"] = probs[0, 1].item()
        else:
            seg_results[task] = {
                "label": state.dataset_metadata[task][idx_pred],
                "confidence": conf,
            }

    stroke = seg_results.get("stroke_type", {"label": "Other", "confidence": 0.0})
    if stroke["confidence"] < STROKE_CONFIDENCE_FLOOR:
        stroke = {"label": "Other", "confidence": stroke["confidence"]}

    return {
        "label": stroke["label"],
        "confidence": stroke["confidence"],
        "badminton_prob": seg_results.get("badminton_prob", 1.0),
        "metrics": {
            "subtype": seg_results.get("stroke_subtype", {"label": "None", "confidence": 0.0}),
            "technique": seg_results.get("technique", {"label": "Unknown", "confidence": 0.0}),
            "placement": seg_results.get("placement", {"label": "Unknown", "confidence": 0.0}),
            "position": seg_results.get("position", {"label": "Unknown", "confidence": 0.0}),
            "intent": seg_results.get("intent", {"label": "None", "confidence": 0.0}),
            "quality": seg_results.get("quality_label", "Developing"),
        },
    }


def _run_pose_check(frames_rgb: list[np.ndarray]) -> float:
    """Pose-based badminton gate on a sample of frames. Blocking."""
    landmarks_list = []
    for frame_rgb in frames_rgb:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        p_results = state.pose_estimator.process_frame(frame_bgr)
        landmarks = state.pose_estimator.get_landmarks_as_list(p_results)
        landmarks_list.append(landmarks[0] if landmarks else [])
    _, pose_conf, _ = state.badminton_detector.is_badminton_video(landmarks_list, threshold=0.0)
    return pose_conf

import time

_commentary_cache: dict[str, str] = {}
_COMMENTARY_CACHE_MAX = 128
_gemini_backoff_until: float = 0.0
_GEMINI_BACKOFF_SECONDS = 30

def _result_signature(result: dict) -> str:
    m = result.get("metrics", {})

    def lab(key):
        value = m.get(key, {})
        if isinstance(value, dict):
            return value.get("label", "?")
        return value or "?"

    band = "low" if float(result.get("confidence") or 0) < 0.5 else "ok"
    return (
        f"{result.get('label', '?')}|{lab('subtype')}|{lab('technique')}|"
        f"{lab('placement')}|{lab('position')}|{lab('intent')}|{m.get('quality', '?')}|{band}"
    )


_LIVE_COMMENTARY_MAX_WORDS = 8
_LIVE_COMMENTARY_GAP_S = 8.0


def _live_commentary_acceptable(text: str) -> bool:
    """At most 8 words; one complete sentence ending in . ! or ?"""
    t = text.strip()
    if not t or t[-1] not in ".!?":
        return False
    return len(t.split()) <= _LIVE_COMMENTARY_MAX_WORDS


async def _generate_commentary(result: dict) -> str:
    """One short Gemini cue. Silence when the model cannot give one."""
    global _gemini_backoff_until

    sig = _result_signature(result)
    if sig in _commentary_cache:
        cached = _commentary_cache[sig]
        if _live_commentary_acceptable(cached):
            return cached
        del _commentary_cache[sig]

    if not state.gemini_enabled or not state.gemini_client:
        return ""

    if time.monotonic() < _gemini_backoff_until:
        return ""

    try:
        prompt = live_coaching_prompt(result, _LIVE_COMMENTARY_MAX_WORDS)
        resp = await asyncio.to_thread(
            state.gemini_client.models.generate_content,
            model=state.gemini_model_name,
            contents=prompt,
        )

        text = ""
        if resp.candidates:
            candidate = resp.candidates[0]
            if candidate.content and candidate.content.parts:
                text = plain_coaching_text(resp.text).split("\n", 1)[0].strip()

        if not text:
            reason = "no_candidates"
            if resp.candidates:
                reason = getattr(resp.candidates[0], "finish_reason", "unknown")
            print(f"[live] gemini: empty response ({reason})")
            return ""

        if not _live_commentary_acceptable(text):
            n = len(text.split())
            print(
                f"[live] gemini: skipped tip (not ≤{_LIVE_COMMENTARY_MAX_WORDS} words "
                f"or incomplete sentence; got {n} words)"
            )
            return ""

        if len(_commentary_cache) >= _COMMENTARY_CACHE_MAX:
            _commentary_cache.pop(next(iter(_commentary_cache)))
        _commentary_cache[sig] = text
        return text
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            _gemini_backoff_until = time.monotonic() + _GEMINI_BACKOFF_SECONDS
            print(f"[live] gemini: rate limited — backing off {_GEMINI_BACKOFF_SECONDS}s")
        elif "503" in msg or "UNAVAILABLE" in msg:
            _gemini_backoff_until = time.monotonic() + _GEMINI_BACKOFF_SECONDS
            print(f"[live] gemini: 503 unavailable — backing off {_GEMINI_BACKOFF_SECONDS}s")
        else:
            print(f"[live] gemini exception: {e}")
        return ""


@router.websocket("/sessions/{session_id}/ws")
async def live_session_ws(ws: WebSocket, session_id: str):
    """
    WebSocket live analysis — frame ingestion and inference are decoupled.
    """
    async with state._live_lock:
        if session_id not in state._live_sessions:
            await ws.close(code=4004, reason="Unknown session")
            return

    await ws.accept()
    print(f"[live] session {session_id} — WebSocket accepted, waiting for frames")
    await ws.send_json({"event": "status", "message": "Warming up — collecting initial frames..."})

    ring: list[np.ndarray] = []
    ring_lock = asyncio.Lock()
    bg_tasks: set[asyncio.Task] = set()
    last_commentary_sig: str | None = None
    last_commentary_at = 0.0
    pending_result: dict | None = None
    closed = asyncio.Event()
    frame_count = 0

    async def _send_commentary(result: dict):
        try:
            text = await _generate_commentary(result)
            if text:
                print(f"[live] gemini: {text[:80]}")
                await ws.send_json({"event": "commentary", "text": text})
            else:
                print("[live] gemini: (empty response)")
        except Exception as e:
            print(f"[live] commentary error: {e}")

    async def _inference_loop():
        nonlocal last_commentary_sig, last_commentary_at, pending_result
        first_sent = False
        was_break = False
        cycle = 0
        pose_passed = False
        while not closed.is_set():
            await asyncio.sleep(INFERENCE_INTERVAL_S)
            if closed.is_set():
                break
            try:
                async with ring_lock:
                    n = len(ring)
                    if n < RING_SIZE:
                        continue
                    snapshot = list(ring[-RING_SIZE:])

                if not first_sent:
                    print(f"[live] first inference — ring has {n} frames")
                    await ws.send_json({"event": "status", "message": "Analyzing..."})
                    first_sent = True

                run_pose = (cycle == 0) or (cycle % POSE_CHECK_INTERVAL == 0) or (was_break and not pose_passed)
                if run_pose:
                    step = max(1, len(snapshot) // POSE_SAMPLE_COUNT)
                    pose_sample = snapshot[::step][:POSE_SAMPLE_COUNT]
                    pose_conf = await asyncio.to_thread(_run_pose_check, pose_sample)
                    pose_passed = pose_conf >= POSE_CONF_MIN
                    if not pose_passed:
                        if not was_break:
                            print(f"[live] not badminton — pose gate failed (pose_conf={pose_conf:.3f})")
                            await ws.send_json({
                                "event": "break",
                                "reason": "no_badminton",
                                "message": "No badminton activity detected — point your camera at the court.",
                            })
                            was_break = True
                        cycle += 1
                        continue

                result = await asyncio.to_thread(_run_window_inference, snapshot)

                if result["badminton_prob"] < BADMINTON_PROB_MIN:
                    if not was_break:
                        print(f"[live] break detected (badminton_prob={result['badminton_prob']:.2f})")
                        await ws.send_json({
                            "event": "break",
                            "reason": "game_break",
                            "message": "Break in play detected — waiting for action to resume.",
                        })
                        was_break = True
                    cycle += 1
                    continue

                was_break = False
                await ws.send_json({"event": "analysis", **result})

                if live_cue_worth_saying(result):
                    pending_result = result
                if (
                    pending_result is not None
                    and time.monotonic() - last_commentary_at >= _LIVE_COMMENTARY_GAP_S
                ):
                    sig = _result_signature(pending_result)
                    cue = pending_result
                    pending_result = None
                    if sig != last_commentary_sig:
                        last_commentary_sig = sig
                        last_commentary_at = time.monotonic()
                        task = asyncio.create_task(_send_commentary(cue))
                        bg_tasks.add(task)
                        task.add_done_callback(bg_tasks.discard)
            except Exception as e:
                print(f"[live] inference error: {e}")
            cycle += 1

    inference_task = asyncio.create_task(_inference_loop())

    try:
        while True:
            raw = await ws.receive_bytes()
            bgr = _decode_frame(raw)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (224, 224))
            async with ring_lock:
                ring.append(rgb_resized)
                frame_count += 1
                if len(ring) > RING_SIZE * 2:
                    ring[:] = ring[-RING_SIZE:]
            if frame_count in (1, RING_SIZE, RING_SIZE * 2):
                print(f"[live] frames received: {frame_count}")
    except WebSocketDisconnect:
        print(f"[live] client disconnected ({frame_count} frames received)")
    except Exception as e:
        print(f"[live] receive error: {e}")
        try:
            await ws.send_json({"event": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        closed.set()
        inference_task.cancel()
        for t in bg_tasks:
            t.cancel()
        print(f"[live] session {session_id} cleaned up ({frame_count} total frames)")
        async with state._live_lock:
            if session_id in state._live_sessions:
                del state._live_sessions[session_id]
                state._live_semaphore.release()
