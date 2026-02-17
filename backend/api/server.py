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
from dataset import FineBadmintonDataset
from pose_utils import PoseEstimator
from badminton_detector import BadmintonPoseDetector

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
# Prioritize the backend/models directory where training saves
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/badminton_model.pth")

if not os.path.exists(MODEL_PATH):
    # Fallback to root models directory if present
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/badminton_model.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
pose_estimator = None
badminton_detector = None
dataset_metadata = None

@app.on_event("startup")
async def load_model():
    global model, pose_estimator, badminton_detector, dataset_metadata
    
    # Initialize metadata from dataset class
    dummy_root = os.path.join(os.path.dirname(__file__), "../data")
    dummy_json = os.path.join(dummy_root, "transformed_combined_rounds_output_en_evals_translated.json")
    
    try:
        temp_dataset = FineBadmintonDataset(dummy_root, dummy_json)
        dataset_metadata = temp_dataset.classes
    except Exception as e:
        print(f"Warning: Could not load dataset metadata: {e}. Using fallback.")
        dataset_metadata = {
            "stroke_type": ["Serve", "Clear", "Smash", "Drop", "Drive", "Net_Shot", "Lob", "Defensive_Shot", "Other"],
            "stroke_subtype": ["None", "Short_Serve", "Flick_Serve", "High_Serve", "Common_Smash", "Jump_Smash", "Full_Smash", "Stick_Smash", "Slice_Smash", "Slice_Drop", "Stop_Drop", "Reverse_Slice_Drop", "Blocked_Drop", "Flat_Lift", "High_Lift", "Net_Lift", "Attacking_Clear", "Spinning_Net", "Flat_Drive", "High_Drive", "Other"],
            "technique": ["Forehand", "Backhand", "Turnaround", "Unknown"],
            "placement": ["Straight", "Cross-court", "Body_Hit", "Over_Head", "Passing_Shot", "Wide", "Unknown"],
            "position": ["Mid_Front", "Mid_Court", "Mid_Back", "Left_Front", "Left_Mid", "Left_Back", "Right_Front", "Right_Mid", "Right_Back", "Unknown"],
            "intent": ["Intercept", "Passive", "Defensive", "To_Create_Depth", "Move_To_Net", "Early_Net_Shot", "Deception", "Hesitation", "Seamlessly", "None"]
        }

    task_classes = {k: len(v) for k, v in dataset_metadata.items()}
    task_classes["quality"] = 7
    
    # Read hidden_size from model registry to match saved checkpoint
    import json
    hidden_size = 128 # default
    registry_path = os.path.join(os.path.dirname(MODEL_PATH), "model_registry.json")
    if os.path.exists(registry_path):
        try:
            with open(registry_path) as f:
                registry = json.load(f)
            model_name = os.path.basename(MODEL_PATH)
            if model_name in registry.get("models", {}):
                hidden_size = registry["models"][model_name].get("hidden_size", 128)
                acc = registry["models"][model_name].get("accuracy", "?")
                print(f"Registry: Loading {model_name} (accuracy={acc}%, hidden_size={hidden_size})")
        except Exception as e:
            print(f"Warning: Could not read model registry: {e}")
    
    model = CNN_LSTM_Model(task_classes=task_classes, hidden_size=hidden_size)
    
    if os.path.exists(MODEL_PATH):
        abs_path = os.path.abspath(MODEL_PATH)
        print(f"Attempting to load model from: {abs_path}")
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"SUCCESS: Model loaded from {abs_path}.")
        except Exception as e:
            print(f"ERROR: Failed to load state_dict: {e}")
            sys.exit(1)
    else:
        print(f"CRITICAL: Model file NOT found at {os.path.abspath(MODEL_PATH)}.")
        sys.exit(1)

    
    model.to(device)
    model.eval()
    
    pose_estimator = PoseEstimator()
    badminton_detector = BadmintonPoseDetector()
    print("Pose Estimator and Badminton Detector initialized.")

def process_segment(cap, start_frame, end_frame, sequence_length=16):
    """Extract frames and pose for a specific video segment."""
    frames = []
    indices = np.linspace(start_frame, end_frame, sequence_length).astype(int)
    
    pose_img = None
    middle_idx = len(indices) // 2
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            if i == middle_idx:
                p_results = pose_estimator.process_frame(frame)
                annotated = pose_estimator.draw_landmarks(frame, p_results)
                h, w = annotated.shape[:2]
                scale = 320 / max(h, w)
                pose_img = cv2.resize(annotated, (int(w*scale), int(h*scale)))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_resized)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
    frames_np = np.array(frames)
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0)
    
    encoded_pose = ""
    if pose_img is not None:
        _, buffer = cv2.imencode('.jpg', pose_img)
        encoded_pose = base64.b64encode(buffer).decode('utf-8')
        
    return frames_tensor, encoded_pose

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    import uuid
    # Use UUID to avoid filename collisions and race conditions
    file_id = str(uuid.uuid4())
    temp_file = f"temp_{file_id}_{file.filename}"
    
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    cap = cv2.VideoCapture(temp_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # NEW: Read all frames sequentially to avoid CAP_PROP_POS_FRAMES issues on macOS
    all_frames_rgb = []
    all_frames_raw = [] # Keep original for pose images
    
    print(f"Buffering video frames for {file.filename} (ID: {file_id})...")
    sys.stdout.flush()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames_raw.append(frame.copy())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        all_frames_rgb.append(frame_resized)
        if len(all_frames_rgb) % 60 == 0:
            print(f" -> Buffered {len(all_frames_rgb)} frames...")
            sys.stdout.flush()
    cap.release()
    
    total_frames = len(all_frames_rgb)
    print(f"Buffering complete. Total frames: {total_frames}")
    sys.stdout.flush()
    
    if total_frames == 0:
        if os.path.exists(temp_file): os.remove(temp_file)
        return {"error": "Could not read video frames", "validation_failed": True}
    
    # STAGE 1: MediaPipe Pose-Based Validation
    print("Stage 1: Analyzing poses for badminton characteristics...")
    sys.stdout.flush()
    
    # Sample frames for pose analysis (every 10th frame, max 30 frames)
    sample_indices = list(range(0, total_frames, max(1, total_frames // 30)))[:30]
    pose_landmarks_list = []
    
    for idx in sample_indices:
        raw_frame = all_frames_raw[idx]
        p_results = pose_estimator.process_frame(raw_frame)
        landmarks = pose_estimator.get_landmarks_as_list(p_results)
        if landmarks:
            pose_landmarks_list.append(landmarks[0])  # First person detected
        else:
            pose_landmarks_list.append([])
    
    is_badminton_pose, pose_confidence, pose_details = badminton_detector.is_badminton_video(
        pose_landmarks_list, threshold=0.05  # Very permissive - only reject obvious non-badminton
    )
    
    print(f"Pose Analysis: is_badminton={is_badminton_pose}, confidence={pose_confidence:.3f}")
    print(f"Details: {pose_details}")
    sys.stdout.flush()
    
    # If pose analysis strongly suggests not badminton, reject early
    # Only reject if confidence is EXTREMELY low (< 0.05)
    if not is_badminton_pose and pose_confidence < 0.05:
        if os.path.exists(temp_file): os.remove(temp_file)
        return {
            "error": "This doesn't appear to be a badminton video. Please upload a video showing badminton gameplay with visible players and racket movements.",
            "validation_failed": True,
            "validation_details": {
                "stage": "pose_analysis",
                "pose_confidence": pose_confidence,
                "overhead_score": pose_details.get("overhead_score", 0.0),
                "racket_score": pose_details.get("racket_score", 0.0),
                "stance_score": pose_details.get("stance_score", 0.0)
            }
        }
        
    timeline = []
    
    try:
        # 1. Sliding Window Analysis
        # Window size: 1.5s, Step: 0.75s
        window_size_frames = int(1.5 * fps)
        step_size_frames = int(0.75 * fps)
        
        # Ensure we have at least one window
        if total_frames < window_size_frames:
            window_size_frames = total_frames
            step_size_frames = total_frames
            
        print(f"Starting sliding window analysis (FPS: {fps:.2f})...")
        sys.stdout.flush()
        for start in range(0, total_frames - window_size_frames // 2, step_size_frames):
            end = min(start + window_size_frames, total_frames)
            timestamp = f"{int(start/fps)//60:02d}:{int(start/fps)%60:02d}"
            
            indices = np.linspace(start, end - 1, 16).astype(int)
            
            # Extract 16 frames for this segment
            segment_frames = [all_frames_rgb[idx] for idx in indices]
            segment_tensor = torch.from_numpy(np.array(segment_frames)).float() / 255.0
            segment_tensor = segment_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)
            
            # Extract pose image from the middle of the segment
            middle_idx = indices[len(indices)//2]
            raw_frame = all_frames_raw[middle_idx]
            
            
            p_results = pose_estimator.process_frame(raw_frame)
            annotated = pose_estimator.draw_landmarks(raw_frame, p_results)
            h, w = annotated.shape[:2]
            scale = 320 / max(h, w)
            pose_img = cv2.resize(annotated, (int(w*scale), int(h*scale)))
            _, buffer = cv2.imencode('.jpg', pose_img)
            pose_b64 = base64.b64encode(buffer).decode('utf-8')
            
            with torch.no_grad():
                outputs = model(segment_tensor)
                
                
                seg_results = {}
                for task, logits in outputs.items():
                    probs = torch.softmax(logits, dim=1)
                    idx = torch.argmax(probs, dim=1).item()
                    conf = probs[0, idx].item()
                    
                    if task == "quality":
                        # Map 7 model classes to 10-point scale for better granularity
                        # 1->1-2, 2->3, 3->4-5, 4->6, 5->7-8, 6->9, 7->10
                        quality_map_to_10 = {
                            0: 1,   # Developing -> 1-2
                            1: 3,   # Emerging -> 3
                            2: 5,   # Competent -> 4-5
                            3: 6,   # Proficient -> 6
                            4: 8,   # Advanced -> 7-8
                            5: 9,   # Expert -> 9
                            6: 10   # Elite -> 10
                        }
                        seg_results["quality_numeric"] = quality_map_to_10.get(idx, 5)
                        
                        # Updated labels for 10-point scale
                        q_map = {
                            1: "Beginner", 2: "Beginner+", 3: "Developing", 
                            4: "Competent", 5: "Competent+", 6: "Proficient",
                            7: "Advanced", 8: "Advanced+", 9: "Expert", 10: "Elite"
                        }
                        seg_results["quality_label"] = q_map.get(seg_results["quality_numeric"], "Unknown")
                    elif task == "is_badminton":
                        # Store is_badminton prediction for validation
                        seg_results["is_badminton_conf"] = conf
                        seg_results["is_badminton_pred"] = idx  # 0=not_badminton, 1=badminton
                    else:
                        seg_results[task] = {
                            "label": dataset_metadata[task][idx],
                            "confidence": conf
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
                        "quality": seg_results.get("quality_label", "Developing")
                    }
                })

        print(f"Analysis complete. Found {len(timeline)} segments.")
        sys.stdout.flush()
        
        # STAGE 2: Model-Based Binary Classification Validation
        print("Stage 2: Validating badminton classification from model...")
        sys.stdout.flush()
        
        # Calculate average is_badminton confidence across all segments
        badminton_confidences = [seg.get("is_badminton_conf", 0.0) for seg in timeline]
        badminton_predictions = [seg.get("is_badminton_pred", 0) for seg in timeline]
        
        avg_badminton_conf = np.mean(badminton_confidences) if badminton_confidences else 0.0
        badminton_ratio = sum(badminton_predictions) / len(badminton_predictions) if badminton_predictions else 0.0
        
        print(f"Model Classification: avg_confidence={avg_badminton_conf:.3f}, badminton_ratio={badminton_ratio:.3f}")
        sys.stdout.flush()
        
        # Combined validation: require either high pose score OR high model confidence
        # This allows flexibility for different video qualities
        pose_passed = pose_confidence >= 0.30
        model_passed = badminton_ratio >= 0.5 and avg_badminton_conf >= 0.50
        
        if not (pose_passed or model_passed):
            if os.path.exists(temp_file): os.remove(temp_file)
            
            # Provide specific feedback based on what failed
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
                    "model_passed": model_passed
                }
            }
        
        print("Validation passed! Proceeding with full analysis...")
        sys.stdout.flush()

        # 2. Pick Global Best
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
            # Updated reverse mapping for 10-point scale
            q_rev = {
                "Beginner": 1, "Beginner+": 2, "Developing": 3, 
                "Competent": 4, "Competent+": 5, "Proficient": 6,
                "Advanced": 7, "Advanced+": 8, "Expert": 9, "Elite": 10
            }
            # If label not found (e.g. old labels), default to 5 (Competent+)
            numeric_quality = q_rev.get(quality_label, 5)

            # Apply weighted quality scoring (User requested 10% weight for pose)
            # Old: Hard caps based on pose -> Harsh penalty
            # New: Weighted average -> Soft penalty for bad form
            
            # Map pose_confidence (0-1) to 0-10 scale
            pose_rating = pose_confidence * 10
            
            # Weighted average: 90% Model Quality, 10% Pose Quality
            weighted_quality = (numeric_quality * 0.9) + (pose_rating * 0.1)
            numeric_quality = int(round(weighted_quality))
            
            # Add note if form score is low but we're not harshly penalizing
            # User requested to remove detailed flags for cleaner UI
            # if pose_confidence < 0.3:
            #      if "(Form Adjusted)" not in quality_label:
            #         quality_label = f"{quality_label} (Form Adjusted)"

            # Keep Model Confidence Penalty (Prediction Certainty)
            # If the model itself is unsure, we still cap the score
            if confidence < 0.3:
                numeric_quality = min(numeric_quality, 4)
                # if "(Low Confidence)" not in quality_label:
                #     quality_label = f"{quality_label} (Low Confidence)"
            elif confidence < 0.5:
                numeric_quality = min(numeric_quality, 7)

            # 2. Model Confidence Penalty (Prediction Certainty)
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

        # 3. Recommendations (Gemini)
        # ... (stays the same, but handle nested dict for action_label/subtype)
        
        reco_subtype = metrics.get('subtype', {}).get('label', 'None') if isinstance(metrics.get('subtype'), dict) else 'None'
        reco_tech = metrics.get('technique', {}).get('label', 'Unknown') if isinstance(metrics.get('technique'), dict) else 'Unknown'
        reco_place = metrics.get('placement', {}).get('label', 'Unknown') if isinstance(metrics.get('placement'), dict) else 'Unknown'
        reco_pos = metrics.get('position', {}).get('label', 'Unknown') if isinstance(metrics.get('position'), dict) else 'Unknown'
        reco_intent = metrics.get('intent', {}).get('label', 'None') if isinstance(metrics.get('intent'), dict) else 'None'

        recommendations = []
        if gemini_enabled:
            try:
                tactical_context = (
                    f"- Stroke: {action_label} ({reco_subtype})\n"
                    f"- Technique: {reco_tech}\n"
                    f"- Placement: {reco_place}\n"
                    f"- Court Position: {reco_pos}\n"
                    f"- Tactical Intent: {reco_intent}\n"
                    f"- Quality: {numeric_quality}/10 ({quality_label})"
                )
                prompt = (
                    f"You are a professional badminton coach. Analyze this stroke data and provide 3 concise technical tips:\n"
                    f"{tactical_context}\n\n"
                    f"Format as single-line bullet points."
                )
                response = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
                recommendations = [t.strip().lstrip('*-•').strip() for t in response.text.strip().split('\n') if t.strip()][:3]
            except:
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
            "timeline": timeline
        }
            
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.get("/")
def read_root():
    return {"message": "Badminton Coach API is running!"}
