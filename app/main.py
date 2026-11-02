import asyncio
import base64
import hashlib
import hmac
import json
import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiomqtt

from database import (
    init_db,
    get_db_stats,
    get_connection,
    IMAGES_DIR,
    MAX_IMAGES_PER_HOUR,
    update_person_details,
    merge_persons,
    delete_person,
    delete_hourly_image,
    get_custom_tags,
    upsert_custom_tag,
    delete_custom_tag,
)
from face_engine import face_engine
from mqtt_worker import mqtt_listener_loop, mqtt_stats, MQTT_BROKER_HOST, MQTT_BROKER_PORT
from email_alerter import send_alert_email, get_alert_settings

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme_admin")
GUEST_USERNAME = os.getenv("GUEST_USERNAME", "guest")
GUEST_PASSWORD = os.getenv("GUEST_PASSWORD", "changeme_guest")
SESSION_SECRET = os.getenv("SESSION_SECRET", "axis_face_secret_key_change_me_2026")
COOKIE_NAME = "axis_face_session"


def create_session_token(username: str, role: str) -> str:
    payload = f"{username}:{role}"
    sig = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def verify_session_token(token: Optional[str]) -> Optional[dict]:
    if not token or token.count(":") != 2:
        return None
    username, role, sig = token.split(":", 2)
    expected = hmac.new(
        SESSION_SECRET.encode(), f"{username}:{role}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    return {"username": username, "role": role, "is_admin": role == "admin"}


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(COOKIE_NAME)
    return verify_session_token(token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    face_engine.start()
    mqtt_task = asyncio.create_task(mqtt_listener_loop())
    yield
    mqtt_task.cancel()


app = FastAPI(title="Axis Facial Recognition Server", lifespan=lifespan)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
templates = Jinja2Templates(directory="/app/templates")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    u = username.strip()
    if u == ADMIN_USERNAME and hmac.compare_digest(password, ADMIN_PASSWORD):
        token = create_session_token(u, "admin")
    elif u == GUEST_USERNAME and hmac.compare_digest(password, GUEST_PASSWORD):
        token = create_session_token(u, "guest")
    else:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password"},
            status_code=401,
        )

    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie(key=COOKIE_NAME, value=token, httponly=True, samesite="lax", max_age=86400 * 7)
    return resp


@app.post("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp


@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    tab: str = "faces",
    gallery_id: Optional[int] = None,
    sort_by: str = "last_seen_desc",
    std_tag_filter: str = "",
    custom_tag_filter: str = "",
    person_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    flash: Optional[str] = None,
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    conn = get_connection()
    try:
        sort_map = {
            "last_seen_desc": "p.last_seen DESC",
            "std_tag_asc": "p.person_type ASC, p.last_seen DESC",
            "std_tag_desc": "p.person_type DESC, p.last_seen DESC",
            "custom_tag_asc": "CASE WHEN p.custom_tag IS NULL OR p.custom_tag = '' THEN 1 ELSE 0 END ASC, p.custom_tag ASC, p.last_seen DESC",
            "custom_tag_desc": "CASE WHEN p.custom_tag IS NULL OR p.custom_tag = '' THEN 1 ELSE 0 END ASC, p.custom_tag DESC, p.last_seen DESC",
            "name_asc": "p.name ASC",
            "sightings_desc": "p.total_sightings DESC, p.last_seen DESC",
        }
        order_sql = sort_map.get(sort_by, "p.last_seen DESC")

        where_clauses = ["1=1"]
        where_params = []
        if std_tag_filter in ("Known", "Unknown"):
            where_clauses.append("p.person_type = ?")
            where_params.append(std_tag_filter)
        if custom_tag_filter == "__none__":
            where_clauses.append("(p.custom_tag IS NULL OR p.custom_tag = '')")
        elif custom_tag_filter:
            where_clauses.append("p.custom_tag = ?")
            where_params.append(custom_tag_filter)

        where_sql = " AND ".join(where_clauses)

        persons = conn.execute(
            f"""
            SELECT p.*,
                   COALESCE(c.friendly_name, p.last_camera) AS camera_friendly_name,
                   (SELECT COUNT(*) FROM hourly_images h WHERE h.person_id = p.id) AS stored_images_count
            FROM persons p
            LEFT JOIN cameras c ON p.last_camera = c.camera_id
            WHERE {where_sql}
            ORDER BY {order_sql}
            """,
            where_params,
        ).fetchall()

        all_persons_for_merge = conn.execute(
            "SELECT id, name FROM persons ORDER BY name ASC"
        ).fetchall()

        cameras = conn.execute("SELECT * FROM cameras ORDER BY camera_id ASC").fetchall()
        custom_tags = get_custom_tags()

        gallery_person = None
        gallery_images = []
        if gallery_id:
            gallery_person = conn.execute("SELECT * FROM persons WHERE id = ?", (gallery_id,)).fetchone()
            gallery_images = conn.execute(
                "SELECT * FROM hourly_images WHERE person_id = ? ORDER BY captured_at DESC",
                (gallery_id,),
            ).fetchall()

        sightings = []
        filter_p_id = int(person_id) if (person_id and person_id.isdigit()) else None
        if tab == "history":
            query = """
                SELECT s.*, p.name AS person_name, p.person_type, p.custom_tag, p.best_image_path,
                       COALESCE(c.friendly_name, s.camera_id) AS camera_friendly_name
                FROM sightings s
                JOIN persons p ON s.person_id = p.id
                LEFT JOIN cameras c ON s.camera_id = c.camera_id
                WHERE 1=1
            """
            params = []
            if filter_p_id:
                query += " AND s.person_id = ?"
                params.append(filter_p_id)
            if camera_id:
                query += " AND s.camera_id = ?"
                params.append(camera_id)
            query += " ORDER BY s.captured_at DESC LIMIT 200"
            sightings = conn.execute(query, params).fetchall()

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "is_admin": user["is_admin"],
                "active_tab": tab,
                "persons": persons,
                "all_persons": all_persons_for_merge,
                "cameras": cameras,
                "custom_tags": custom_tags,
                "sightings": sightings,
                "gallery_person": gallery_person,
                "gallery_images": gallery_images,
                "sort_by": sort_by,
                "std_tag_filter": std_tag_filter,
                "custom_tag_filter": custom_tag_filter,
                "filter_person_id": filter_p_id,
                "filter_camera_id": camera_id,
                "cfg": get_alert_settings(),
                "gpu": face_engine.gpu_status,
                "mqtt": mqtt_stats,
                "stats": get_db_stats(),
                "max_images_per_hour": MAX_IMAGES_PER_HOUR,
                "flash": flash,
            },
        )
    finally:
        conn.close()


@app.post("/ui/person/{person_id}/update")
async def ui_update_person(
    request: Request,
    person_id: int,
    name: str = Form(...),
    person_type: str = Form(...),
    custom_tag: str = Form(""),
    alert_rule: str = Form("default"),
    sort_by: str = Form("last_seen_desc"),
    std_tag_filter: str = Form(""),
    custom_tag_filter: str = Form(""),
):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=faces&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    update_person_details(person_id, name, person_type, alert_rule, custom_tag)
    return RedirectResponse(
        url=f"/?tab=faces&sort_by={sort_by}&std_tag_filter={std_tag_filter}&custom_tag_filter={custom_tag_filter}&flash=Saved+person+details",
        status_code=303,
    )


@app.post("/ui/image/{image_id}/delete")
async def ui_delete_hourly_image(
    request: Request,
    image_id: int,
    person_id: int = Form(...),
    sort_by: str = Form("last_seen_desc"),
    std_tag_filter: str = Form(""),
    custom_tag_filter: str = Form(""),
):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=faces&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    delete_hourly_image(image_id)
    return RedirectResponse(
        url=f"/?tab=faces&gallery_id={person_id}&sort_by={sort_by}&std_tag_filter={std_tag_filter}&custom_tag_filter={custom_tag_filter}&flash=Image+permanently+deleted#person-row-{person_id}",
        status_code=303,
    )


@app.post("/ui/tags/upsert")
async def ui_upsert_custom_tag(
    request: Request,
    name: str = Form(...),
    alert_enabled: str = Form("false"),
):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=settings&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    upsert_custom_tag(name=name, alert_enabled=(alert_enabled == "true"))
    return RedirectResponse(url="/?tab=settings&flash=Custom+tag+saved", status_code=303)


@app.post("/ui/tags/delete")
async def ui_delete_custom_tag(request: Request, name: str = Form(...)):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=settings&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    delete_custom_tag(name=name)
    return RedirectResponse(url="/?tab=settings&flash=Custom+tag+deleted", status_code=303)


@app.post("/ui/person/{person_id}/merge")
async def ui_merge_person(request: Request, person_id: int, target_id: int = Form(...)):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=faces&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    merge_persons(source_id=person_id, target_id=target_id)
    return RedirectResponse(url="/?tab=faces&flash=Merged+profiles+successfully", status_code=303)


@app.post("/ui/person/{person_id}/delete")
async def ui_delete_person(request: Request, person_id: int):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=faces&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    delete_person(person_id)
    return RedirectResponse(url="/?tab=faces&flash=Deleted+person", status_code=303)


@app.post("/ui/settings/email")
async def ui_save_email_settings(
    request: Request,
    alert_enabled: str = Form("false"),
    alert_on_known: str = Form("true"),
    alert_on_unknown: str = Form("true"),
    alert_cooldown_minutes: str = Form("15"),
    smtp_host: str = Form(""),
    smtp_port: str = Form("587"),
    smtp_username: str = Form(""),
    smtp_password: str = Form(""),
    alert_from_email: str = Form(""),
    alert_to_email: str = Form(""),
):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=settings&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    updates = {
        "alert_enabled": alert_enabled,
        "alert_on_known": alert_on_known,
        "alert_on_unknown": alert_on_unknown,
        "alert_cooldown_minutes": alert_cooldown_minutes,
        "smtp_host": smtp_host.strip(),
        "smtp_port": smtp_port.strip(),
        "smtp_username": smtp_username.strip(),
        "smtp_password": smtp_password,
        "alert_from_email": alert_from_email.strip(),
        "alert_to_email": alert_to_email.strip(),
    }
    conn = get_connection()
    try:
        with conn:
            for k, v in updates.items():
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, v),
                )
    finally:
        conn.close()
    return RedirectResponse(url="/?tab=settings&flash=Email+settings+saved", status_code=303)


@app.post("/ui/settings/test-email")
async def ui_test_email(request: Request):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=settings&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    cfg = get_alert_settings()
    try:
        await send_alert_email(
            person_name="Test Person",
            person_type="Known",
            custom_tag="VIP",
            camera_name="Web UI Test",
            captured_at="Now",
            quality_score=0.95,
            similarity=0.90,
            image_bytes=None,
            cfg=cfg,
        )
        return RedirectResponse(url="/?tab=settings&flash=Test+email+sent+successfully", status_code=303)
    except Exception as e:
        return RedirectResponse(url=f"/?tab=settings&flash=Email+Error:+{str(e)}", status_code=303)


@app.post("/ui/camera/update")
async def ui_update_camera(
    request: Request, camera_id: str = Form(...), friendly_name: str = Form(...)
):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse(url="/?tab=settings&flash=Permission+denied+(Guest+is+view-only)", status_code=303)

    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE cameras SET friendly_name = ? WHERE camera_id = ?",
                (friendly_name.strip(), camera_id),
            )
    finally:
        conn.close()
    return RedirectResponse(url="/?tab=settings&flash=Camera+renamed", status_code=303)


@app.get("/api/status")
async def api_status():
    return {
        "gpu_status": face_engine.gpu_status,
        "mqtt_status": mqtt_stats,
        "db_status": get_db_stats(),
        "max_images_per_hour": MAX_IMAGES_PER_HOUR,
    }


@app.post("/reset-db")
async def reset_db(request: Request):
    user = get_current_user(request)
    if not user or not user["is_admin"]:
        return JSONResponse({"error": "Admin privileges required"}, status_code=403)

    conn = get_connection()
    with conn:
        conn.executescript(
            """
            DELETE FROM sightings;
            DELETE FROM hourly_images;
            DELETE FROM embeddings;
            DELETE FROM persons;
            DELETE FROM sqlite_sequence WHERE name IN ('persons', 'embeddings', 'hourly_images', 'sightings');
            """
        )
    conn.close()
    if IMAGES_DIR.exists():
        for child in IMAGES_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    return {"reset": True, "db_status": get_db_stats()}