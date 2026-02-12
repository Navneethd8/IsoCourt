from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import torch
import numpy as np
import sys
import base64
from google import genai
from dotenv import load_dotenv

# Add parent directory to path to import model and dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import CNN_LSTM_Model
from pose_utils import PoseEstimator

# Load and validate environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
    print("WARNING: GEMINI_API_KEY not found or not set in .env. Falling back to static coaching tips.")
    gemini_enabled = False
else:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # Use Gemini 3.0 Flash Preview
        model_name = 'gemini-3-flash-preview'
        gemini_enabled = True
        print(f"SUCCESS: {model_name} enabled via google-genai SDK.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Gemini: {e}")
        gemini_enabled = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model (Global variable)
MODEL_PATH = "../models/badminton_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
pose_estimator = None

@app.on_event("startup")
async def load_model():
    global model, pose_estimator
    # Initialize model architecture
    model = CNN_LSTM_Model(num_classes=11, num_quality_classes=7)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Warning: Model file not found. Using untrained model for inference.")
    
    model.to(device)
    model.eval()
    
    # Initialize Pose Estimator
    pose_estimator = PoseEstimator()
    print("Pose Estimator initialized.")

def preprocess_video(video_path, sequence_length=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Simple uniform sampling
    indices = np.linspace(0, total_frames-1, sequence_length).astype(int)
    
    # To store the frame with best pose for visualization
    middle_frame_idx = indices[len(indices)//2]
    best_pose_image = None
    
    for count, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Check for pose on the middle frame for visualization
            if count == len(indices)//2:
                # Process original frame for pose (or a copy)
                results = pose_estimator.process_frame(frame)
                annotated_frame = pose_estimator.draw_landmarks(frame, results)
                
                # Resize for display (max dimension 640 to keep size reasonable but quality good)
                h, w = annotated_frame.shape[:2]
                scale = 640 / max(h, w)
                if scale < 1:
                    new_w, new_h = int(w * scale), int(h * scale)
                    best_pose_image = cv2.resize(annotated_frame, (new_w, new_h))
                else:
                    best_pose_image = annotated_frame

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    
    frames_np = np.array(frames)
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0
    # (T, H, W, C) -> (batch=1, T, C, H, W)
    frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0)
    
    return frames_tensor, best_pose_image

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # Define globally within function for all blocks to use
    classes = [
        "Serve", "Forehand_Clear", "Backhand_Clear", "Smash", "Drop", 
        "Drive", "Net_Shot", "Lob", "Defensive_Shot", "Other", "Unknown"
    ]
    quality_labels = ["Score 1", "Score 2", "Score 3", "Score 4", "Score 5", "Score 6", "Score 7"]
    
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Preprocess to get input tensor for classification
        input_tensor, _ = preprocess_video(temp_file)
        input_tensor = input_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            action_logits, quality_logits = model(input_tensor)
            action_prob = torch.softmax(action_logits, dim=1)
            
            action_idx = torch.argmax(action_prob, dim=1).item()
            confidence = action_prob[0, action_idx].item()
            quality_prob = torch.softmax(quality_logits, dim=1)
            quality_idx = torch.argmax(quality_prob, dim=1).item()
            
            # Mapping
            action_label = classes[action_idx] if action_idx < len(classes) else "Unknown"
            
            # Enhanced Quality Mapping (1-7 scale)
            # 1-2: Emerging, 3-4: Competent, 5-6: Advanced, 7: Elite
            quality_descriptions = {
                1: "Developing", 2: "Emerging", 3: "Competent", 
                4: "Proficient", 5: "Advanced", 6: "Expert", 7: "Elite"
            }
            numeric_quality = quality_idx + 1
            quality_label = quality_descriptions.get(numeric_quality, "Unknown")

            # Coaching Recommendation Logic
            recommendations = []
            
            # Static Fallback Data
            static_tips = {
                "Smash": ["Focus on a higher contact point for better down-angle.", "Engage your core and rotate your hips more for power."],
                "Clear": ["Drive your elbow higher to get better depth.", "Keep your wrist loose until the point of impact."],
                "Drop": ["Disguise the shot by using the same preparation.", "Focus on a softer touch and 'brushing' the shuttle."],
                "Serve": ["Keep your stance stable.", "For short serves, aim to just clear the net tape."],
                "Net_Shot": ["Keep your racket head high.", "Use a gentle 'pushing' motion."],
                "Default": ["Focus on early preparation.", "Maintain a balanced 'ready position'."]
            }

            tactical_analysis = {
                "handedness": "Unknown",
                "direction": "Unknown",
                "intent": "Unknown",
                "specific_type": action_label
            }

            if gemini_enabled:
                try:
                    prompt = (
                        f"You are a world-class professional badminton coach. "
                        f"Analyze this player's stroke:\n"
                        f"- Stroke Type: {action_label}\n"
                        f"- Model Confidence: {confidence:.2f}\n"
                        f"- Execution Quality Score: {numeric_quality}/7 (7 is Elite, 1 is Beginner)\n\n"
                        f"Provide the following in a structured format:\n"
                        f"1. TACTICAL_METRICS: [Forehand/Backhand], [Straight/Cross-court/Body hit], [Specific Subtype], [Strategy/Intent]\n"
                        f"2. COACH_TIPS: 2-3 specific, concise technical cues.\n\n"
                        f"Format the metrics as a single comma-separated line after 'TACTICAL_METRICS:'. "
                        f"Format tips as single-line bullet points after 'COACH_TIPS:'."
                    )
                    response = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
                    text = response.text.strip()
                    
                    # Parse Tactical Metrics
                    if "TACTICAL_METRICS:" in text:
                        metrics_line = text.split("TACTICAL_METRICS:")[1].split('\n')[0].strip()
                        parts = [p.strip() for p in metrics_line.split(',')]
                        if len(parts) >= 4:
                            tactical_analysis["handedness"] = parts[0]
                            tactical_analysis["direction"] = parts[1]
                            tactical_analysis["specific_type"] = parts[2]
                            tactical_analysis["intent"] = parts[3]

                    # Parse Coach Tips
                    if "COACH_TIPS:" in text:
                        tips_section = text.split("COACH_TIPS:")[1]
                        raw_tips = tips_section.strip().split('\n')
                        recommendations = [t.strip().lstrip('*-•').strip() for t in raw_tips if t.strip()][:3]
                    else:
                        recommendations = static_tips.get(action_label, static_tips["Default"])

                    print(f"SUCCESS: Generated detailed tactical analysis via Gemini 3.0 Flash for {action_label}.")
                except Exception as e:
                    print(f"WARNING: Gemini generation failed: {e}. Using static fallbacks.")
                    recommendations = static_tips.get(action_label, static_tips["Default"])
            else:
                # Use static tips if Gemini not enabled
                recommendations = static_tips.get(action_label, static_tips["Default"])

            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(action_prob, 3)
            top3_actions = []
            for i in range(3):
                idx = top3_indices[0, i].item()
                label = classes[idx] if idx < len(classes) else "Unknown"
                score = top3_prob[0, i].item()
                top3_actions.append({"label": label, "score": score})

            # Check video duration
            cap = cv2.VideoCapture(temp_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            timeline = []
            
            if duration > 1.2:
                print(f"Video detected ({duration:.2f}s, {frame_count} frames). Running sliding window analysis...")
                # Sliding window parameters
                window_size = 1.2 # seconds
                stride = 0.6 # seconds
                
                # Open video once for the whole timeline analysis
                cap = cv2.VideoCapture(temp_file)
                
                current_time = 0.0
                print(f"  Inference Start (Device: {device})")
                
                
                while current_time + window_size <= duration:
                    segment_pose_b64 = None
                    # Extract frames for this window
                    start_frame = int(current_time * fps)
                    end_frame = int((current_time + window_size) * fps)
                    end_frame = min(end_frame, frame_count)
                    
                    snippet_frames = []
                    # 1.2s window. We need 16 frames.
                    frames_to_extract = end_frame - start_frame
                    step = max(1, frames_to_extract // 16)
                    
                    for i in range(16):
                        frame_idx = start_frame + i * step
                        if frame_idx >= end_frame: break
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            if i == 8:
                                try:
                                    results = pose_estimator.process_frame(frame)
                                    annotated_frame = pose_estimator.draw_landmarks(frame, results)
                                    h, w = annotated_frame.shape[:2]
                                    scale = 320 / max(h, w)
                                    segment_pose_img = cv2.resize(annotated_frame, (int(w*scale), int(h*scale))) if scale < 1 else annotated_frame
                                    _, buffer = cv2.imencode('.jpg', segment_pose_img)
                                    segment_pose_b64 = base64.b64encode(buffer).decode('utf-8')
                                except Exception as e:
                                    print(f"    [Pose Error] {e}")
                            
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = cv2.resize(frame, (224, 224))
                            snippet_frames.append(frame)
                        else:
                            snippet_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                    
                    while len(snippet_frames) < 16:
                        snippet_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                        
                    # Prepare tensor and inference
                    snippet_np = np.array(snippet_frames)
                    snippet_tensor = torch.from_numpy(snippet_np).float() / 255.0
                    snippet_tensor = snippet_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)
                    # No normalization matching dataset.py training (div 255 only)
                    
                    with torch.no_grad():
                        s_action_logits, _ = model(snippet_tensor)
                        s_action_prob = torch.softmax(s_action_logits, dim=1)
                        s_top3_prob, s_top3_indices = torch.topk(s_action_prob, 3)
                        
                        s_action_idx = s_top3_indices[0, 0].item()
                        s_confidence = s_top3_prob[0, 0].item()
                        s_label = classes[s_action_idx] if s_action_idx < len(classes) else "Unknown"
                        
                        debug_top3 = ", ".join([f"{classes[s_top3_indices[0,k].item()]}: {s_top3_prob[0,k].item():.4f}" for k in range(3)])
                        print(f"  Window {start_frame}-{end_frame} ({current_time:.2f}s): {debug_top3}")


                        if s_confidence > 0.1:
                             timeline.append({
                                 "timestamp": f"{int(current_time // 60):02d}:{int(current_time % 60):02d}",
                                 "label": s_label,
                                 "confidence": s_confidence,
                                 "pose_image": segment_pose_b64
                             })
                    
                    current_time += stride
                cap.release()

            print(f"Generated Timeline: {timeline}")

            # Standard analysis (full video)
            return {
                "action": action_label,
                "confidence": float(action_prob[0, action_idx].item()),
                "top_3_actions": top3_actions,
                "quality": quality_label,
                "quality_numeric": numeric_quality,
                "quality_score": float(quality_prob[0, quality_idx].item()),
                "recommendations": recommendations,
                "tactical_analysis": tactical_analysis,
                "timeline": timeline if timeline else None
            }
            
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.get("/")
def read_root():
    return {"message": "Badminton Coach API is running!"}
