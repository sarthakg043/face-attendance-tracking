import base64
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from config import TEMP_DIR
from database import db_insert_attendance, db_get_last_event_today, db_get_all_persons
from face_engine import match_face, extract_embedding, _best_distance
from config import EUCLIDEAN_THR, COSINE_THR

router = APIRouter(prefix="/attendance", tags=["attendance"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@router.get("")
async def attendance_page(request: Request):
    return templates.TemplateResponse(request, "attendance.html")


@router.post("/mark")
async def mark_attendance(request: Request):
    """
    Expects JSON: { photo: base64_jpeg, gps_lat?, gps_lon? }
    Auto-detects check_in / check_out based on last event today.
    Records IP address and GPS location.
    """
    body = await request.json()
    b64 = body.get("photo", "")
    if not b64:
        return JSONResponse({"success": False, "error": "No photo provided."}, 400)

    # Extract client IP
    ip_address = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not ip_address:
        ip_address = request.client.host if request.client else None

    # GPS from client
    gps_lat = body.get("gps_lat")
    gps_lon = body.get("gps_lon")

    # Save to temp
    img_data = base64.b64decode(b64.split(",")[-1])
    tmp_path = str(TEMP_DIR / f"{uuid.uuid4().hex}.jpg")
    try:
        with open(tmp_path, "wb") as f:
            f.write(img_data)

        result = match_face(tmp_path, top_k=3)

        if not result["is_match"]:
            # Check if face belongs to a deactivated student
            inactive = [p for p in db_get_all_persons(active_only=False) if not p.get("active", 1)]
            if inactive and result.get("_probe_emb") is None:
                # Re-extract probe embedding to check against inactive
                probe_emb = extract_embedding(tmp_path)
                for person in inactive:
                    euc, cos = _best_distance(probe_emb, person["embedding"])
                    if euc < EUCLIDEAN_THR and cos < COSINE_THR:
                        return JSONResponse({
                            "success": False,
                            "deactivated": True,
                            "error": f"Your account ({person['name']} — {person['roll_no']}) has been deactivated. Please contact the admin.",
                        })

            return JSONResponse({
                "success": False,
                "error": "Face not recognised. Please try again or contact admin.",
                "strategy": result["strategy_comparison"],
            })

        person = result["best_match"]

        # Auto-detect event
        last_event = db_get_last_event_today(person["roll_no"])
        if last_event is None or last_event == "check_out":
            event = "check_in"
        else:
            event = "check_out"

        score = {"euclidean": person["euclidean"], "cosine": person["cosine"]}
        att_id = db_insert_attendance(
            person["person_id"], person["roll_no"], person["name"], event, score,
            ip_address=ip_address, gps_lat=gps_lat, gps_lon=gps_lon,
        )

        return JSONResponse({
            "success": True,
            "attendance_id": att_id,
            "roll_no": person["roll_no"],
            "name": person["name"],
            "department": person["department"],
            "event": event,
            "match_score": score,
        })
    except ValueError as e:
        err_msg = str(e)
        if "Face could not be detected" in err_msg:
            return JSONResponse({
                "success": False,
                "error": "No face detected in the photo. Please ensure your face is clearly visible, well-lit, and centered in the frame.",
            }, 422)
        return JSONResponse({"success": False, "error": "Could not process the photo. Please try again."}, 422)
    except Exception:
        return JSONResponse({"success": False, "error": "Something went wrong. Please try again."}, 500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
