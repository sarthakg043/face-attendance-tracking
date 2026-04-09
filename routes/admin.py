import csv
import io
import json
import base64
import os
import uuid
from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fpdf import FPDF

from auth import verify_admin, require_admin
from config import PHOTO_DIR, TEMP_DIR
from database import (
    db_get_all_persons,
    db_get_person_by_roll,
    db_update_person,
    db_delete_person,
    db_set_person_active,
    db_get_attendance_by_date,
    db_get_attendance_by_person,
    db_get_attendance_range,
    db_delete_attendance,
    get_daily_summary,
    db_insert_person,
)
from face_engine import extract_embedding

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


# ── Auth ──────────────────────────────────────────────────────────────────────

@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")


@router.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if verify_admin(username, password):
        request.session["admin"] = True
        return RedirectResponse("/admin/dashboard", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": "Invalid credentials."})


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/admin/login", status_code=302)


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/dashboard")
@require_admin
async def dashboard(request: Request):
    persons = db_get_all_persons()
    summary = get_daily_summary()
    return templates.TemplateResponse(request, "admin/dashboard.html", {
        "persons": persons,
        "summary": summary,
        "today": date.today().isoformat(),
    })


# ── Student Management ────────────────────────────────────────────────────────

@router.get("/students")
@require_admin
async def students_list(request: Request):
    persons = db_get_all_persons()
    return templates.TemplateResponse(request, "admin/students.html", {
        "persons": persons,
    })


@router.get("/students/{roll_no}")
@require_admin
async def student_detail(request: Request, roll_no: str):
    person = db_get_person_by_roll(roll_no)
    if not person:
        return RedirectResponse("/admin/students", status_code=302)
    history = db_get_attendance_by_person(roll_no)
    return templates.TemplateResponse(request, "admin/student_detail.html", {
        "person": person,
        "history": history,
    })


@router.get("/students/{roll_no}/photo")
@require_admin
async def student_photo(request: Request, roll_no: str):
    person = db_get_person_by_roll(roll_no)
    if person and person.get("photo_path") and os.path.exists(person["photo_path"]):
        return FileResponse(person["photo_path"], media_type="image/jpeg")
    return JSONResponse({"error": "No photo"}, 404)


@router.post("/students/{roll_no}/toggle")
@require_admin
async def toggle_student(request: Request, roll_no: str):
    person = db_get_person_by_roll(roll_no)
    if person:
        db_set_person_active(roll_no, not bool(person.get("active", 1)))
    return RedirectResponse(f"/admin/students/{roll_no}", status_code=302)


@router.post("/students/{roll_no}/update")
@require_admin
async def update_student(request: Request, roll_no: str,
                         name: str = Form(...), department: str = Form(...)):
    db_update_person(roll_no, name=name, department=department)
    return RedirectResponse(f"/admin/students/{roll_no}", status_code=302)


@router.post("/students/{roll_no}/delete")
@require_admin
async def delete_student(request: Request, roll_no: str):
    db_delete_person(roll_no, remove_photo=True)
    return RedirectResponse("/admin/students", status_code=302)


@router.post("/students/{roll_no}/re-enroll")
@require_admin
async def re_enroll_student(request: Request, roll_no: str):
    """Re-enroll via JSON body with base64 photos."""
    body = await request.json()
    photos = body.get("photos", [])
    person = db_get_person_by_roll(roll_no)
    if not person:
        return JSONResponse({"success": False, "error": "Not found."}, 404)

    embeddings = []
    for b64 in photos[:12]:
        try:
            img_data = base64.b64decode(b64.split(",")[-1])
            tmp_path = str(TEMP_DIR / f"{uuid.uuid4().hex}.jpg")
            with open(tmp_path, "wb") as f:
                f.write(img_data)
            emb = extract_embedding(tmp_path)
            embeddings.append(emb)
            os.remove(tmp_path)
        except Exception:
            pass

    if len(embeddings) < 3:
        return JSONResponse({"success": False, "error": "Need at least 3 valid face photos."}, 422)

    stored_photo = str(PHOTO_DIR / f"{roll_no}.jpg")
    db_update_person(roll_no, embedding=embeddings, photo_path=stored_photo)
    return JSONResponse({"success": True, "photos_used": len(embeddings)})


# ── Enrollment from admin ─────────────────────────────────────────────────────

@router.get("/enroll")
@require_admin
async def admin_enroll_page(request: Request):
    return templates.TemplateResponse(request, "admin/enroll.html")


# ── Reports ───────────────────────────────────────────────────────────────────

@router.get("/reports")
@require_admin
async def reports_page(request: Request):
    query_date = request.query_params.get("date", date.today().isoformat())
    summary = get_daily_summary(query_date)
    records = db_get_attendance_by_date(query_date)
    return templates.TemplateResponse(request, "admin/reports.html", {
        "summary": summary,
        "records": records,
        "query_date": query_date,
    })


@router.get("/reports/export/csv")
@require_admin
async def export_csv(request: Request):
    start = request.query_params.get("start", date.today().isoformat())
    end = request.query_params.get("end", date.today().isoformat())
    records = db_get_attendance_range(start, end)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Roll No", "Name", "Event", "Timestamp", "Date", "IP Address", "GPS Lat", "GPS Lon", "Match Score"])
    for r in records:
        writer.writerow([r["id"], r["roll_no"], r["name"], r["event"],
                         r["timestamp"], r["date"],
                         r.get("ip_address", ""), r.get("gps_lat", ""), r.get("gps_lon", ""),
                         r.get("match_score", "")])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{start}_to_{end}.csv"},
    )


@router.get("/reports/export/pdf")
@require_admin
async def export_pdf(request: Request):
    start = request.query_params.get("start", date.today().isoformat())
    end = request.query_params.get("end", date.today().isoformat())
    records = db_get_attendance_range(start, end)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Attendance Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Period: {start} to {end}", ln=True, align="C")
    pdf.ln(5)

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    col_w = [15, 30, 40, 25, 45, 25]
    headers = ["#", "Roll No", "Name", "Event", "Timestamp", "Date"]
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for idx, r in enumerate(records, 1):
        pdf.cell(col_w[0], 7, str(idx), border=1, align="C")
        pdf.cell(col_w[1], 7, r["roll_no"], border=1)
        pdf.cell(col_w[2], 7, r["name"], border=1)
        pdf.cell(col_w[3], 7, r["event"].replace("_", " "), border=1, align="C")
        pdf.cell(col_w[4], 7, r["timestamp"], border=1)
        pdf.cell(col_w[5], 7, r["date"], border=1, align="C")
        pdf.ln()

    pdf_bytes = pdf.output()
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=attendance_{start}_to_{end}.pdf"},
    )


@router.post("/attendance/{att_id}/delete")
@require_admin
async def delete_att_record(request: Request, att_id: int):
    db_delete_attendance(att_id)
    referer = request.headers.get("referer", "/admin/reports")
    return RedirectResponse(referer, status_code=302)
