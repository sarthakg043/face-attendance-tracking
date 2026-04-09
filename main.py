from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from config import SECRET_KEY, BASE_DIR
from database import init_database
from routes.enrollment import router as enrollment_router
from routes.attendance import router as attendance_router
from routes.admin import router as admin_router

app = FastAPI(title="Face Recognition Attendance System")

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(enrollment_router)
app.include_router(attendance_router)
app.include_router(admin_router)

# ── Templates ─────────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
async def startup():
    init_database()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


if __name__ == "__main__":
    import uvicorn
    from config import PORT
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
