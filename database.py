import sqlite3
import json
import os
from datetime import datetime, date
from config import DB_PATH, PHOTO_DIR


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_database() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS persons (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        roll_no     TEXT    UNIQUE NOT NULL,
        name        TEXT    NOT NULL,
        department  TEXT    NOT NULL,
        photo_path  TEXT,
        embedding   TEXT    NOT NULL,
        active      INTEGER NOT NULL DEFAULT 1,
        enrolled_at TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS attendance (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id   INTEGER NOT NULL REFERENCES persons(id),
        roll_no     TEXT    NOT NULL,
        name        TEXT    NOT NULL,
        event       TEXT    NOT NULL CHECK(event IN ('check_in','check_out')),
        timestamp   TEXT    NOT NULL,
        date        TEXT    NOT NULL,
        match_score TEXT,
        ip_address  TEXT,
        gps_lat     REAL,
        gps_lon     REAL
    );
    """
    with get_connection() as conn:
        conn.executescript(ddl)
        # Add 'active' column to existing DBs that lack it
        cols = [r[1] for r in conn.execute("PRAGMA table_info(persons)").fetchall()]
        if "active" not in cols:
            conn.execute("ALTER TABLE persons ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        # Add location columns to existing DBs
        att_cols = [r[1] for r in conn.execute("PRAGMA table_info(attendance)").fetchall()]
        if "ip_address" not in att_cols:
            conn.execute("ALTER TABLE attendance ADD COLUMN ip_address TEXT")
        if "gps_lat" not in att_cols:
            conn.execute("ALTER TABLE attendance ADD COLUMN gps_lat REAL")
        if "gps_lon" not in att_cols:
            conn.execute("ALTER TABLE attendance ADD COLUMN gps_lon REAL")


# ── Row helper ────────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    d = dict(row)
    if "embedding" in d and d["embedding"]:
        parsed = json.loads(d["embedding"])
        if parsed and isinstance(parsed[0], (int, float)):
            d["embedding"] = [parsed]
        else:
            d["embedding"] = parsed
    return d


# ── PERSONS CRUD ──────────────────────────────────────────────────────────────

def db_insert_person(roll_no: str, name: str, department: str,
                     photo_path: str, embedding: list) -> int:
    sql = """
        INSERT INTO persons (roll_no, name, department, photo_path, embedding, active, enrolled_at)
        VALUES (?, ?, ?, ?, ?, 1, ?)
    """
    with get_connection() as conn:
        try:
            cur = conn.execute(sql, (
                roll_no, name, department, photo_path,
                json.dumps(embedding),
                datetime.now().isoformat(timespec="seconds"),
            ))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"Roll number '{roll_no}' is already enrolled.")


def db_get_person_by_roll(roll_no: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM persons WHERE roll_no = ?", (roll_no,)).fetchone()
    return _row_to_dict(row) if row else None


def db_get_all_persons(active_only: bool = False) -> list[dict]:
    sql = "SELECT * FROM persons"
    if active_only:
        sql += " WHERE active = 1"
    sql += " ORDER BY id"
    with get_connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [_row_to_dict(r) for r in rows]


def db_get_person_by_id(person_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    return _row_to_dict(row) if row else None


def db_update_person(roll_no: str, **fields) -> bool:
    allowed = {"name", "department", "photo_path", "embedding", "active"}
    updates = {}
    for k, v in fields.items():
        if k not in allowed:
            raise ValueError(f"Field '{k}' cannot be updated.")
        updates[k] = json.dumps(v) if k == "embedding" else v
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    sql = f"UPDATE persons SET {set_clause} WHERE roll_no = ?"
    with get_connection() as conn:
        cur = conn.execute(sql, (*updates.values(), roll_no))
    return cur.rowcount > 0


def db_delete_person(roll_no: str, remove_photo: bool = False) -> bool:
    person = db_get_person_by_roll(roll_no)
    if not person:
        return False
    with get_connection() as conn:
        conn.execute("DELETE FROM attendance WHERE roll_no = ?", (roll_no,))
        conn.execute("DELETE FROM persons WHERE roll_no = ?", (roll_no,))
    if remove_photo and person.get("photo_path") and os.path.exists(person["photo_path"]):
        os.remove(person["photo_path"])
    return True


def db_set_person_active(roll_no: str, active: bool) -> bool:
    return db_update_person(roll_no, active=int(active))


# ── ATTENDANCE CRUD ───────────────────────────────────────────────────────────

def db_insert_attendance(person_id: int, roll_no: str, name: str,
                         event: str, match_score: dict | None = None,
                         ip_address: str | None = None,
                         gps_lat: float | None = None,
                         gps_lon: float | None = None) -> int:
    now = datetime.now()
    sql = """
        INSERT INTO attendance (person_id, roll_no, name, event, timestamp, date, match_score, ip_address, gps_lat, gps_lon)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (
            person_id, roll_no, name, event,
            now.isoformat(timespec="seconds"),
            now.date().isoformat(),
            json.dumps(match_score) if match_score else None,
            ip_address, gps_lat, gps_lon,
        ))
    return cur.lastrowid


def db_get_last_event_today(roll_no: str) -> str | None:
    today = date.today().isoformat()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT event FROM attendance WHERE roll_no = ? AND date = ? ORDER BY timestamp DESC LIMIT 1",
            (roll_no, today),
        ).fetchone()
    return row["event"] if row else None


def db_get_attendance_by_date(query_date: str | None = None) -> list[dict]:
    if query_date is None:
        query_date = date.today().isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM attendance WHERE date = ? ORDER BY timestamp", (query_date,)
        ).fetchall()
    return [dict(r) for r in rows]


def db_get_attendance_by_person(roll_no: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM attendance WHERE roll_no = ? ORDER BY timestamp", (roll_no,)
        ).fetchall()
    return [dict(r) for r in rows]


def db_get_attendance_range(start_date: str, end_date: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM attendance WHERE date BETWEEN ? AND ? ORDER BY date, timestamp",
            (start_date, end_date),
        ).fetchall()
    return [dict(r) for r in rows]


def db_delete_attendance(attendance_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM attendance WHERE id = ?", (attendance_id,))
    return cur.rowcount > 0


def get_daily_summary(query_date: str | None = None) -> dict:
    if query_date is None:
        query_date = date.today().isoformat()
    records = db_get_attendance_by_date(query_date)
    timelines: dict[str, dict] = {}
    for r in records:
        rn = r["roll_no"]
        if rn not in timelines:
            timelines[rn] = {"name": r["name"], "events": []}
        timelines[rn]["events"].append({"event": r["event"], "time": r["timestamp"]})

    summary_rows = []
    for rn, info in timelines.items():
        events = info["events"]
        check_in = next((e["time"] for e in events if e["event"] == "check_in"), None)
        check_out = next((e["time"] for e in reversed(events) if e["event"] == "check_out"), None)
        duration_min = None
        if check_in and check_out:
            t_in = datetime.fromisoformat(check_in)
            t_out = datetime.fromisoformat(check_out)
            duration_min = round((t_out - t_in).total_seconds() / 60, 1)
        summary_rows.append({
            "roll_no": rn,
            "name": info["name"],
            "check_in": check_in,
            "check_out": check_out,
            "duration_min": duration_min,
            "status": "complete" if (check_in and check_out)
                      else "checked_in" if check_in else "absent",
        })

    return {
        "date": query_date,
        "total_present": len([s for s in summary_rows if s["status"] != "absent"]),
        "total_complete": len([s for s in summary_rows if s["status"] == "complete"]),
        "persons": summary_rows,
    }
