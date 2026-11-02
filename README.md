# 🏸 IsoCourt: AI-Native Badminton Analysis

IsoCourt is a state-of-the-art action recognition and coaching platform designed to analyze badminton footage, provide play-by-play breakdowns, and deliver expert technical tips using deep learning and Large Language Models.

---

## 🏗 System Overview

```text
       ResNet + LSTM
             ↓
  Biomechanics Rule Engine
             ↓
 Structured Insight Summary
             ↓
          Gemini
```

IsoCourt operates as a continuous **Sliding Window** pipeline. The backend loops through the video in 1.5s segments with a 0.75s overlap, ensuring every frame is processed with temporal context and high-fidelity coaching feedback.

### 1. The Frontend (Vite + React)
Users upload their badminton footage through a Vite-powered React SPA (`frontend/`). The app handles large file transfers and provides a real-time **Match Timeline**. Once analysis is complete, it renders a synchronized view of action labels, confidence scores, and frame-by-frame **Skeleton Analysis**. SEO uses static `public/robots.txt` / `sitemap.xml` (rewritten on build when `VITE_SITE_URL` is set), plus per-route head updates via `src/seo/`.

### 2. The Backend (FastAPI)
The heart of the system processes the video through a sophisticated pipeline:
- **Frame Buffering**: Loads video segments into memory to ensure stability and high-speed access.
- **Action Recognition**: Uses the **Architecture V2 CNN-LSTM** model to identify the stroke type, player position, and technique from 16-frame sequences.
- **Pose Estimation**: Simultaneously runs MediaPipe to extract skeleton data for visual feedback.

### 3. The Intelligence Layer (Google Gemini)
The raw data from the model (e.g., "Backhand Clear, Mid-Back, Quality 4/7") is sent to **Gemini 1.5 Flash**. The LLM interprets these metrics to provide three personalized, technical coaching tips for every detected segment.

Paper training, evaluation, and figures live in [JVC](https://github.com/Navneethd8/jvc) (`/Volumes/NewVolume/jvc` on the external drive). This repo is the product. Model classes the API still loads are copied in both places.

---

## 🧠 Deep Learning Engine (Architecture V2)

The core movement analysis is powered by a **CNN-LSTM Hierarchical Model** optimized for temporal badminton actions.

### 1. Feature Extraction (The "Eyes")
- **Base**: ResNet-50 ResNet architecture (frozen or fine-tuned).
- **Function**: Extracts 2048 high-level features per frame.
- **Robustness**: Trained with aggressive `ColorJitter` to be invariant to court color and lighting.

### 2. Temporal Processing (The "Memory")
- **Module**: Dual-layer LSTM with 512 hidden units.
- **Fixed-Frame Analysis**: Samples 16 frames per hitting event using a sliding window.

### 3. Feature Pooling (The "Focus")
- **Global Average Pooling**: Captures the overall context of the swing.
- **Global Max Pooling**: Captures the "peak action" moment of the hit.
- **Concatenation**: Merges both into a 1024-dimension vector for top-tier classification accuracy.

### 4. Multi-Task Heads
A single pass provides 7 distinct layers of analysis:
- **Stroke Type**: Smash, Clear, Drop, etc.
- **Court Position**: Mid-Court, Left-Back, Right-Front, etc.
- **Technique**: Forehand vs. Backhand.
- **Quality**: Performance score (1-7).
- **Tactical Intent**: Deception, Passive, Defensive, etc.

---

## ⚡️ Backend Pipeline

1.  **Buffered Processing**: Videos are read frame-by-frame on the server to prevent memory spikes and bypass macOS metadata race conditions.
2.  **Sliding Window**: Analysis happens in 1.5s windows with 0.75s overlaps, ensuring no action is missed between frames.
3.  **Pose Synchronization**: MediaPipe Pose estimation is run in parallel to provide visual "Skeleton Analysis" for every detected hit.
4.  **LLM Coaching**: The model's raw data is fed into **Google Gemini**, which acts as a virtual "Pro Coach" to provide 3 concise technical tips per segment.
