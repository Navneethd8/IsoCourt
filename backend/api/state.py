"""Process-wide singletons populated at startup (see api.lifespan)."""
import asyncio
from cachetools import TTLCache

from api import config

device = "cpu"
model = None
model_architecture = "cnn_lstm"
pose_estimator = None
badminton_detector = None
dataset_metadata = None

inference_dataset_id = ""
inference_dataset_source = ""
inference_frames_policy = ""
inference_data_root = ""
inference_list_file = ""

gemini_enabled = False
gemini_client = None
gemini_model_name = "gemini-3.1-flash-lite-preview"

# Legacy /analyze + shared cache
_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_JOBS)
_active_jobs = 0
_active_jobs_lock = asyncio.Lock()

_result_cache: TTLCache = TTLCache(maxsize=128, ttl=3600)
_frame_tip_cache: dict = {}

# Live session stub
_live_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LIVE_SESSIONS)
_live_sessions: dict[str, None] = {}
_live_lock = asyncio.Lock()
