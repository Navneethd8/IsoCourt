"""Application startup: load model, pose, detector, Gemini client."""
import os
import sys

import torch
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import state

from core.dataset import FINEBADMINTON_LABELS
from core.pose_utils import PoseEstimator
from core.badminton_detector import BadmintonPoseDetector
from api.model_loader import load_registry, load_stroke_model, resolve_model_path

load_dotenv()

MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../models"))
if not os.path.isdir(MODELS_DIR):
    MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../models"))

MODEL_PATH: str = ""


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    global MODEL_PATH

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "YOUR_API_KEY_HERE":
        print("WARNING: GEMINI_API_KEY not found. Static coaching tips only.")
        state.gemini_enabled = False
        state.gemini_client = None
    else:
        try:
            state.gemini_client = genai.Client(api_key=gemini_key)
            state.gemini_enabled = True
            print(f"SUCCESS: {state.gemini_model_name} enabled via google-genai SDK.")
        except Exception as e:
            print(f"ERROR: Failed to initialize Gemini: {e}")
            state.gemini_enabled = False
            state.gemini_client = None

    registry = load_registry(MODELS_DIR)
    state.dataset_metadata = {k: list(v) for k, v in FINEBADMINTON_LABELS.items()}
    task_classes = {k: len(v) for k, v in state.dataset_metadata.items()}
    task_classes["quality"] = 7

    state.device = "cuda" if torch.cuda.is_available() else "cpu"

    path = resolve_model_path(MODELS_DIR, registry)
    if not path:
        print(f"CRITICAL: No model checkpoint found under {MODELS_DIR}.")
        sys.exit(1)

    MODEL_PATH = path
    abs_path = os.path.abspath(MODEL_PATH)
    print(f"Loading model from: {abs_path}")

    try:
        state.model, arch = load_stroke_model(MODEL_PATH, task_classes, registry, state.device)
        state.model_architecture = arch
        print(f"Architecture: {arch}")
    except Exception as e:
        print(f"ERROR: Failed to build/load model: {e}")
        sys.exit(1)

    state.model.to(state.device)
    state.model.eval()

    state.pose_estimator = PoseEstimator()
    state.badminton_detector = BadmintonPoseDetector()
    print("Pose Estimator and Badminton Detector initialized.")

    yield
