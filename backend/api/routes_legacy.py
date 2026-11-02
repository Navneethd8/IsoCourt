"""Legacy routes: /analyze, /analyze/stream, /health, /frame-tip, /feedback."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from api import state
from api.coaching_text import coaching_lines, frame_tip_prompt, quality_score
from api.clip_analysis import run_analysis_sync, run_analyze_stream_async, sse_line
from api.clip_jobs import clip_manager
from api.config import MAX_CONCURRENT_JOBS, MAX_QUEUED_JOBS
from api.temp_video import safe_unlink, write_video_bytes_to_tempfile

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FEEDBACK_TO_EMAIL = os.getenv("FEEDBACK_TO_EMAIL", "")

# HF Space is the inference API only — keep crawlers off it. Public SEO lives on www.
_ROBOTS_TXT = """\
# IsoCourt API (Hugging Face Space) — not the marketing site
User-agent: *
Disallow: /

# Public site: https://www.isocourt.fit/robots.txt
"""

router = APIRouter(tags=["legacy"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "active_jobs": state._active_jobs,
        "max_jobs": MAX_CONCURRENT_JOBS,
        "queue_available": MAX_CONCURRENT_JOBS - state._active_jobs,
        "clip_queue_depth": clip_manager.clip_queue_depth(),
        "clip_workers_busy": clip_manager.workers_busy(),
        "live_sessions_active": len(state._live_sessions),
        "max_queued_clips": MAX_QUEUED_JOBS,
    }


@router.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt():
    return _ROBOTS_TXT


@router.get("/")
def read_root():
    return {"message": "Badminton Coach API is running!"}


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    content = await file.read()
    video_hash = hashlib.sha256(content).hexdigest()

    if video_hash in state._result_cache:
        cached = dict(state._result_cache[video_hash])
        cached["cache_hit"] = True
        return cached

    if state._semaphore.locked() and state._semaphore._value == 0:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "We're at full capacity right now — IsoCourt is getting a lot of love! Please try again in a moment.",
                "retry_after": 30,
            },
        )

    try:
        await asyncio.wait_for(state._semaphore.acquire(), timeout=0.5)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "We're at full capacity right now — IsoCourt is getting a lot of love! Please try again in a moment.",
                "retry_after": 30,
            },
        )

    async with state._active_jobs_lock:
        state._active_jobs += 1

    try:
        temp_file = write_video_bytes_to_tempfile(content)
    except OSError as e:
        state._semaphore.release()
        async with state._active_jobs_lock:
            state._active_jobs -= 1
        raise HTTPException(status_code=500, detail=f"Failed to store upload: {e}") from e

    try:
        result = await asyncio.to_thread(run_analysis_sync, temp_file)

        if result and not result.get("validation_failed"):
            state._result_cache[video_hash] = result

        result["cache_hit"] = False
        return result

    finally:
        safe_unlink(temp_file)
        state._semaphore.release()
        async with state._active_jobs_lock:
            state._active_jobs -= 1


@router.post("/frame-tip")
async def frame_tip(body: dict):
    stroke = body.get("label", "Unknown")
    subtype = body.get("subtype", "None")
    tech = body.get("technique", "Unknown")
    place = body.get("placement", "Unknown")
    position = body.get("position", "Unknown")
    intent = body.get("intent", "None")
    quality = body.get("quality", "Developing")
    conf = body.get("confidence", 0.0)

    cache_key = f"{stroke}|{subtype}|{tech}|{place}|{position}|{intent}|{quality}"
    if cache_key in state._frame_tip_cache:
        return {"tip": state._frame_tip_cache[cache_key], "cached": True}

    if not state.gemini_enabled or not state.gemini_client:
        fallback = "Focus on early preparation and maintaining a balanced stance."
        return {"tip": fallback, "cached": False}

    try:
        prompt = frame_tip_prompt(
            stroke,
            subtype,
            float(conf or 0),
            {
                "technique": {"label": tech},
                "placement": {"label": place},
                "position": {"label": position},
                "intent": {"label": intent},
            },
            quality,
            quality_score(quality),
        )
        response = await asyncio.to_thread(
            state.gemini_client.models.generate_content,
            model=state.gemini_model_name,
            contents=prompt,
        )
        lines = coaching_lines(response.text, 1)
        tip = lines[0] if lines else ""
        state._frame_tip_cache[cache_key] = tip
        return {"tip": tip, "cached": False}
    except Exception as e:
        print(f"Frame tip error: {e}")
        return {"tip": "Maintain your ready position and prepare early for the next shot.", "cached": False}


@router.post("/analyze/stream")
async def analyze_video_stream(file: UploadFile = File(...)):
    content = await file.read()
    video_hash = hashlib.sha256(content).hexdigest()

    if video_hash in state._result_cache:
        cached = state._result_cache[video_hash]

        async def stream_cached():
            for i, seg in enumerate(cached.get("timeline", [])):
                event = json.dumps({"event": "progress", "window": i, **seg})
                yield f"data: {event}\n\n"
                await asyncio.sleep(0)
            summary = {k: v for k, v in cached.items() if k != "timeline"}
            summary["cache_hit"] = True
            yield f"data: {json.dumps({'event': 'done', 'summary': summary})}\n\n"

        return StreamingResponse(stream_cached(), media_type="text/event-stream")

    if state._semaphore._value <= 0:

        async def at_capacity():
            yield f"data: {json.dumps({'event': 'error', 'error': 'Server at capacity. Please try again shortly.', 'retry_after': 30})}\n\n"

        return StreamingResponse(at_capacity(), media_type="text/event-stream", status_code=503)

    async def generate():
        await state._semaphore.acquire()
        async with state._active_jobs_lock:
            state._active_jobs += 1
        temp_file = ""
        try:
            try:
                temp_file = write_video_bytes_to_tempfile(content)
            except OSError as e:
                yield sse_line({"event": "error", "error": f"Failed to store upload: {e}"})
                return
            async for ev in run_analyze_stream_async(temp_file, video_hash):
                yield sse_line(ev)
        except Exception as e:
            yield sse_line({"event": "error", "error": str(e)})
        finally:
            safe_unlink(temp_file)
            state._semaphore.release()
            async with state._active_jobs_lock:
                state._active_jobs -= 1

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/feedback")
async def send_feedback(body: dict):
    name = (body.get("name") or "").strip()
    email = (body.get("email") or "").strip()
    message = (body.get("message") or "").strip()

    if not name or not email or not message:
        raise HTTPException(status_code=422, detail="Name, email, and message are all required.")

    if not SENDGRID_API_KEY or not FEEDBACK_TO_EMAIL:
        print("WARNING: SENDGRID_API_KEY or FEEDBACK_TO_EMAIL not set. Feedback not sent.")
        raise HTTPException(status_code=503, detail="Feedback service is not configured yet.")

    mail = Mail(
        from_email=FEEDBACK_TO_EMAIL,
        to_emails=FEEDBACK_TO_EMAIL,
        subject=f"[IsoCourt Feedback] from {name}",
        html_content=(
            f"<h3>New feedback from IsoCourt</h3>"
            f"<p><strong>Name:</strong> {name}</p>"
            f"<p><strong>Email:</strong> {email}</p>"
            f"<hr/>"
            f"<p>{message}</p>"
        ),
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(mail)
        return {"ok": True}
    except Exception as e:
        print(f"SendGrid error: {e}")
        raise HTTPException(status_code=500, detail="Failed to send feedback. Please try again later.") from e
