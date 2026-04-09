import base64
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import PHOTO_DIR, TEMP_DIR
from database import db_insert_person, db_get_person_by_roll, db_update_person
from face_engine import extract_embedding
from auth import require_admin

router = APIRouter(prefix="/enroll", tags=["enrollment"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@router.get("")
@require_admin
async def enrollment_page(request: Request):
    return templates.TemplateResponse(request, "enrollment.html")


@router.post("/submit")
@require_admin
async def enrollment_submit(request: Request):
    """
    Expects JSON:
      { roll_no, name, department, photos: [base64_jpeg, ...] }
    Extracts embeddings from each photo, stores in DB.
    """
    body = await request.json()
    roll_no = body.get("roll_no", "").strip()
    name = body.get("name", "").strip()
    department = body.get("department", "").strip()
    photos = body.get("photos", [])

    if not all([roll_no, name, department]):
        return JSONResponse({"success": False, "error": "All fields are required."}, 400)
    if len(photos) < 3:
        return JSONResponse({"success": False, "error": "At least 3 photos are required."}, 400)
    if db_get_person_by_roll(roll_no):
        return JSONResponse({"success": False, "error": f"Roll number '{roll_no}' is already enrolled."}, 409)

    # Save photos to temp, extract embeddings
    embeddings = []
    saved_paths = []
    errors = []

    for i, b64 in enumerate(photos[:12]):
        try:
            img_data = base64.b64decode(b64.split(",")[-1])
            tmp_path = str(TEMP_DIR / f"{uuid.uuid4().hex}.jpg")
            with open(tmp_path, "wb") as f:
                f.write(img_data)
            saved_paths.append(tmp_path)
            emb = extract_embedding(tmp_path)
            embeddings.append(emb)
        except Exception as e:
            errors.append(f"Photo {i+1}: {e}")

    if len(embeddings) < 3:
        # Cleanup temp files
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)
        return JSONResponse({
            "success": False,
            "error": f"Only {len(embeddings)} usable photos (need ≥3). Issues: {'; '.join(errors)}",
        }, 422)

    # Persist first photo as the enrolled photo
    ext = ".jpg"
    stored_photo = str(PHOTO_DIR / f"{roll_no}{ext}")
    if saved_paths:
        os.replace(saved_paths[0], stored_photo)

    # Clean remaining temp files
    for p in saved_paths[1:]:
        if os.path.exists(p):
            os.remove(p)

    person_id = db_insert_person(roll_no, name, department, stored_photo, embeddings)

    return JSONResponse({
        "success": True,
        "person_id": person_id,
        "roll_no": roll_no,
        "name": name,
        "photos_used": len(embeddings),
        "photos_failed": len(errors),
    })


@router.post("/add-photo")
async def add_photo(request: Request):
    """Add extra photo(s) to an existing enrollment."""
    body = await request.json()
    roll_no = body.get("roll_no", "").strip()
    photos = body.get("photos", [])

    person = db_get_person_by_roll(roll_no)
    if not person:
        return JSONResponse({"success": False, "error": "Person not found."}, 404)

    embeddings = person["embedding"]
    added = 0
    for b64 in photos:
        if len(embeddings) >= 12:
            break
        try:
            img_data = base64.b64decode(b64.split(",")[-1])
            tmp_path = str(TEMP_DIR / f"{uuid.uuid4().hex}.jpg")
            with open(tmp_path, "wb") as f:
                f.write(img_data)
            emb = extract_embedding(tmp_path)
            embeddings.append(emb)
            added += 1
            os.remove(tmp_path)
        except Exception:
            pass

    if added:
        db_update_person(roll_no, embedding=embeddings)

    return JSONResponse({
        "success": True,
        "roll_no": roll_no,
        "photos_added": added,
        "total_photos": len(embeddings),
    })
