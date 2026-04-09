import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "attendance.db"
PHOTO_DIR = DATA_DIR / "enrolled_photos"
TEMP_DIR = DATA_DIR / "tmp"

for d in (DATA_DIR, PHOTO_DIR, TEMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── DeepFace ─────────────────────────────────────────────────────────────────
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
EUCLIDEAN_THR = 4.16
COSINE_THR = 0.68

# ── Auth ─────────────────────────────────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(32).hex())
PORT = int(os.getenv("PORT", "8000"))
