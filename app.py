"""
Open Wound Monitoring Dashboard — Backend.

Standalone FastAPI server for open wound patient monitoring.
AI models live inside this app's own "ai model/" folder:
  ai model/detect wound/   — Block 1 wound detection
  ai model/block2/         — Block 2 segmentation & grading
  ai model/block3/         — Block 3 LLM chatbot adapters
Runs on port 8001.

Start:
    lsof -ti:8001 | xargs kill -9 2>/dev/null; ~/anaconda3/envs/block2/bin/python app.py
"""

import io
import json
import os
import secrets
import sys
import threading
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, Form, Depends, HTTPException, UploadFile, File
from fastapi.responses import (
    JSONResponse,
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

BASE_DIR          = Path(__file__).parent
STATIC_DIR        = BASE_DIR / "static"

# Data files
OW_FILE           = BASE_DIR / "data" / "open_wound.json"
OW_IMG_DIR        = BASE_DIR / "data" / "open_wound_sessions"
OW_IMG_DIR.mkdir(parents=True, exist_ok=True)

# AI models — stored inside this app's own "ai model/" folder
AI_MODEL_DIR      = BASE_DIR / "ai model"
BLOCK1_DIR        = AI_MODEL_DIR / "block1"
DETECT_DIR        = AI_MODEL_DIR / "detect wound"
SEG_DIR           = AI_MODEL_DIR / "block2"
CHATBOT_MODELS_DIR = AI_MODEL_DIR / "block3"

# Single YOLO11-seg checkpoint, used for both detection and segmentation.
DETECT_CHECKPOINT = DETECT_DIR / "best.pt"
SEG_CHECKPOINT    = SEG_DIR / "checkpoints" / "best.pt"

USERNAME   = os.environ.get("OW_USER",   "dev team")
PASSWORD   = os.environ.get("OW_PASS",   "12345678")
SECRET_KEY = os.environ.get("OW_SECRET", "ow-change-this-to-a-random-secret")

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}

app = FastAPI(title="Open Wound Monitor")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

_lock = threading.Lock()


# ── Auth ──────────────────────────────────────────────────────────────────────

def is_authed(request: Request) -> bool:
    return request.session.get("authed") is True


def require_auth(request: Request):
    if not is_authed(request):
        raise HTTPException(status_code=401, detail="Not logged in")


LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Login - Open Wound Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{--bg:#f0f4f8;--panel:#ffffff;--border:#e2e8f0;--text:#0f172a;--muted:#64748b;--indigo:#4f46e5;--red:#dc2626;--shadow-sm:0 1px 3px rgba(15,23,42,.07);}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:"Inter",system-ui,sans-serif;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:34px 28px;width:100%;max-width:360px;box-shadow:var(--shadow-sm);}
  h1{font-size:22px;font-weight:700;margin-bottom:4px;letter-spacing:-.02em;}
  h1 .accent{color:var(--indigo);}
  .sub{color:var(--muted);font-size:13px;margin-bottom:22px;}
  label{display:block;font-size:13px;color:var(--muted);margin-bottom:6px;margin-top:14px;}
  label:first-of-type{margin-top:0;}
  input{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:15px;padding:11px 12px;font-family:inherit;}
  input:focus{outline:none;border-color:var(--indigo);}
  button{width:100%;margin-top:18px;background:var(--indigo);color:#ffffff;border:none;border-radius:8px;font-size:15px;font-weight:600;padding:11px;cursor:pointer;font-family:inherit;}
  button:hover{opacity:.9;}
  .err{background:rgba(220,38,38,.08);border:1px solid var(--red);color:var(--red);font-size:13px;padding:10px 12px;border-radius:8px;margin-top:16px;}
</style>
</head>
<body>
  <form class="card" method="post" action="/login">
    <h1>Open <span class="accent">Wound</span> Monitor</h1>
    <div class="sub">Sign in to view and manage patient wound assessments</div>
    {{ERROR}}
    <label for="username">Username</label>
    <input type="text" id="username" name="username" autocomplete="username" autofocus required>
    <label for="password">Password</label>
    <input type="password" id="password" name="password" autocomplete="current-password" required>
    <button type="submit">Sign in</button>
  </form>
</body>
</html>
"""


def render_login(error: bool = False) -> str:
    msg = '<div class="err">Wrong username or password.</div>' if error else ""
    return LOGIN_PAGE.replace("{{ERROR}}", msg)


@app.get("/login", response_class=HTMLResponse)
def login_page(error: int = 0):
    return render_login(error == 1)


@app.post("/login")
def login_submit(request: Request, username: str = Form(""), password: str = Form("")):
    if secrets.compare_digest(username, USERNAME) and secrets.compare_digest(password, PASSWORD):
        request.session["authed"] = True
        return RedirectResponse("/", status_code=303)
    return RedirectResponse("/login?error=1", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/")
def index(request: Request):
    if not is_authed(request):
        return RedirectResponse("/login", status_code=303)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Open wound grade info ─────────────────────────────────────────────────────

_OW_GRADE_INFO = [
    (0, "DTI",         "#6b21a8", "rgba(107,33,168,.2)",  "Suspected deep tissue injury. Off-load pressure immediately. Monitor closely for deterioration."),
    (1, "Unstageable", "#475569", "rgba(71,85,105,.2)",   "Full-thickness skin/tissue loss, depth unknown. Debridement required before staging."),
    (2, "Stage 1",     "#ca8a04", "rgba(202,138,4,.2)",   "Non-blanchable erythema, intact skin. Off-load pressure area. Inspect daily."),
    (3, "Stage 2",     "#ea580c", "rgba(234,88,12,.2)",   "Partial-thickness skin loss. Moist wound healing — appropriate dressing selection. Reposition every 2h."),
    (4, "Stage 3",     "#dc2626", "rgba(220,38,38,.2)",   "Full-thickness skin loss, fat visible. Debridement and advanced dressings required."),
    (5, "Stage 4",     "#9f1239", "rgba(159,18,57,.2)",   "Bone/tendon/muscle exposed. Urgent surgical consult. Osteomyelitis risk — IV antibiotics if indicated."),
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _safe(name: str) -> str:
    return Path(name).name


def _default_ow() -> dict:
    return {
        "patient": {
            "name": "",
            "patient_id": "",
            "wound_onset_date": "",
            "age": 0,
        },
        "sessions": [],
    }


def _load_ow() -> dict:
    if OW_FILE.exists():
        with open(OW_FILE, encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = _default_ow()
        _save_ow(d)
    # Auto-scan session folders for images and videos
    for sess in d.get("sessions", []):
        folder = OW_IMG_DIR / _safe(sess["id"])
        if folder.exists():
            sess["images"] = sorted(
                f.name for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTS
            )
            sess["videos"] = sorted(
                f.name for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS
            )
        else:
            sess["images"] = []
            sess["videos"] = []
    return d


def _save_ow(d: dict):
    OW_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OW_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


# ── Patient ───────────────────────────────────────────────────────────────────

@app.get("/api/demo/data")
def api_demo_data(request: Request):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse(_load_ow())


class DemoPatient(BaseModel):
    name: str = ""
    patient_id: str = ""
    wound_onset_date: str = ""
    age: int = 0


@app.post("/api/demo/patient")
def api_demo_patient(body: DemoPatient, _: None = Depends(require_auth)):
    with _lock:
        d = _load_ow()
        d["patient"] = {
            "name": body.name.strip(),
            "patient_id": body.patient_id.strip(),
            "wound_onset_date": body.wound_onset_date,
            "age": body.age,
        }
        _save_ow(d)
        return JSONResponse({"ok": True, "data": d})


# ── Sessions ──────────────────────────────────────────────────────────────────

class DemoSessionAdd(BaseModel):
    date: str = ""
    grade: int = 0
    wound_width: float = 0.0
    wound_height: float = 0.0
    notes: str = ""


@app.post("/api/demo/session/add")
def api_demo_session_add(body: DemoSessionAdd, _: None = Depends(require_auth)):
    with _lock:
        d = _load_ow()
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        (OW_IMG_DIR / sid).mkdir(parents=True, exist_ok=True)
        sess = {
            "id": sid,
            "date": body.date or datetime.now().strftime("%Y-%m-%d"),
            "grade": max(0, min(5, body.grade)),
            "wound_width": round(body.wound_width, 1),
            "wound_height": round(body.wound_height, 1),
            "notes": body.notes.strip(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "images": [],
        }
        d.setdefault("sessions", []).append(sess)
        _save_ow(d)
        return JSONResponse({"ok": True, "session": sess, "data": d})


@app.delete("/api/demo/session/{sid}")
def api_demo_session_delete(sid: str, _: None = Depends(require_auth)):
    import shutil
    safe_sid = _safe(sid)
    with _lock:
        d = _load_ow()
        d["sessions"] = [s for s in d.get("sessions", []) if s["id"] != safe_sid]
        _save_ow(d)
        p = OW_IMG_DIR / safe_sid
        if p.exists():
            shutil.rmtree(p)
        return JSONResponse({"ok": True, "data": d})


@app.post("/api/demo/session/{sid}/upload")
async def api_demo_session_upload(
    sid: str, request: Request, files: List[UploadFile] = File(...)
):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    safe_sid = _safe(sid)
    with _lock:
        d = _load_ow()
        sess = next((s for s in d.get("sessions", []) if s["id"] == safe_sid), None)
        if not sess:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        sess_dir = OW_IMG_DIR / safe_sid
        sess_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            name = _safe(f.filename or "")
            if not name:
                continue
            content = await f.read()
            with open(sess_dir / name, "wb") as fp:
                fp.write(content)
            if name not in sess.get("images", []):
                sess.setdefault("images", []).append(name)
            saved.append(name)
        _save_ow(d)
    return JSONResponse({"ok": True, "saved": saved, "data": d})


@app.get("/api/demo/session/{sid}/images/{filename}")
def api_demo_session_image(sid: str, filename: str, request: Request):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    path = OW_IMG_DIR / _safe(sid) / _safe(filename)
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/demo/session/{sid}/overlay/{filename}")
def api_demo_session_overlay(sid: str, filename: str, request: Request):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    path = OW_IMG_DIR / _safe(sid) / "overlays" / _safe(filename)
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/demo/session/{sid}/video/{filename}")
def api_demo_session_video(sid: str, filename: str, request: Request):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    path = OW_IMG_DIR / _safe(sid) / _safe(filename)
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)


# ── AI — Model process manager ────────────────────────────────────────────────

import subprocess
import atexit

_running_models: dict[str, subprocess.Popen] = {}


def _kill_all_model_procs():
    for mid, proc in list(_running_models.items()):
        try:
            proc.terminate()
        except Exception:
            pass
    _running_models.clear()


atexit.register(_kill_all_model_procs)

_BLOCK3_PORT_BASE = 8101  # offset from monitor web (which uses 8001+)


def _scan_block3_adapters() -> dict[str, dict]:
    adapters: dict[str, dict] = {}
    try:
        dirs = sorted(
            d for d in CHATBOT_MODELS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    except Exception:
        return adapters

    port = _BLOCK3_PORT_BASE
    for adapter_dir in dirs:
        ac_path = adapter_dir / "adapter_config.json"
        if not ac_path.exists():
            continue
        try:
            with open(ac_path, encoding="utf-8") as f:
                ac = json.load(f)
            base_model = ac.get("base_model_name_or_path") or ac.get("model", "")
            if not base_model:
                continue
        except Exception:
            continue

        model_id = adapter_dir.name
        display_name = model_id.replace("_", " ").title()
        assigned_port = port

        meta_path = adapter_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                display_name = meta.get("name", display_name)
                if "port" in meta:
                    assigned_port = int(meta["port"]) + 100  # offset to avoid conflict
            except Exception:
                pass

        adapters[model_id] = {
            "id": model_id,
            "name": display_name,
            "type": "api",
            "endpoint": f"http://localhost:{assigned_port}",
            "max_tokens": 512,
            "launch": {
                "cmd": [
                    "{block3_python}",
                    str(CHATBOT_MODELS_DIR / "model_server.py"),
                    "--base", base_model,
                    "--adapter", str(adapter_dir),
                    "--name", display_name,
                    "--port", str(assigned_port),
                ]
            },
        }
        port += 1

    return adapters


def _check_endpoint(url: str, timeout: float = 3.0) -> bool:
    import urllib.request as ur
    try:
        with ur.urlopen(url + "/health", timeout=timeout):
            return True
    except Exception:
        return False


@app.get("/api/demo/models")
def api_demo_models(request: Request):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    adapters = _scan_block3_adapters()
    return JSONResponse({"models": list(adapters.values())})


@app.post("/api/demo/model/{model_id}/launch")
def api_launch_model(model_id: str, _: None = Depends(require_auth)):
    adapters = _scan_block3_adapters()
    cfg = adapters.get(_safe(model_id))
    if not cfg:
        return JSONResponse({"ok": False, "error": "Model not found in ai model/block3/"}, status_code=404)

    endpoint = cfg.get("endpoint", "").rstrip("/")
    if _check_endpoint(endpoint, timeout=2.0):
        return JSONResponse({"ok": True, "status": "ready", "endpoint": endpoint})

    proc = _running_models.get(model_id)
    if proc and proc.poll() is None:
        return JSONResponse({"ok": True, "status": "starting", "endpoint": endpoint, "pid": proc.pid})
    if proc:
        del _running_models[model_id]

    launch = cfg.get("launch")
    if not launch or not launch.get("cmd"):
        return JSONResponse({"ok": False, "error": "No launch.cmd in config."}, status_code=400)

    _BLOCK3_PYTHON = str(BASE_DIR.parent / ".venv" / "bin" / "python")

    def _resolve_tok(tok: str) -> str:
        if tok == "{python}":
            return sys.executable
        if tok == "{block3_python}":
            return _BLOCK3_PYTHON
        return tok

    cmd = [_resolve_tok(tok) for tok in launch["cmd"]]
    log_path = BASE_DIR / "data" / f"model_{model_id}.log"
    try:
        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=log_f,
            stderr=log_f,
            close_fds=True,
        )
        _running_models[model_id] = proc
        return JSONResponse({"ok": True, "status": "starting", "endpoint": endpoint, "pid": proc.pid,
                             "log": str(log_path)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to start: {e}"}, status_code=500)


@app.get("/api/demo/model/{model_id}/status")
def api_model_status(model_id: str, _: None = Depends(require_auth)):
    adapters = _scan_block3_adapters()
    cfg = adapters.get(_safe(model_id))
    if not cfg:
        return JSONResponse({"status": "not_found"})

    endpoint = cfg.get("endpoint", "").rstrip("/")
    proc = _running_models.get(model_id)

    if proc and proc.poll() is not None:
        del _running_models[model_id]
        proc = None

    process_managed = proc is not None

    import urllib.request as ur
    try:
        with ur.urlopen(endpoint + "/health", timeout=3) as r:
            health_data = json.load(r)
        return JSONResponse({"status": "ready", "endpoint": endpoint, "managed": process_managed, "health": health_data})
    except Exception:
        return JSONResponse({"status": "starting" if process_managed else "offline",
                             "endpoint": endpoint, "managed": process_managed})


@app.get("/api/demo/model/{model_id}/log")
def api_model_log(model_id: str, _: None = Depends(require_auth)):
    log_path = BASE_DIR / "data" / f"model_{_safe(model_id)}.log"
    if not log_path.exists():
        return JSONResponse({"log": "(no log yet)"})
    return JSONResponse({"log": log_path.read_text(encoding="utf-8", errors="replace")[-4000:]})


@app.post("/api/demo/model/{model_id}/stop")
def api_stop_model(model_id: str, _: None = Depends(require_auth)):
    proc = _running_models.pop(model_id, None)
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass
        return JSONResponse({"ok": True, "message": "Model process stopped"})
    return JSONResponse({"ok": False, "message": "No managed process"})


# ── LLM system prompts ────────────────────────────────────────────────────────

def _build_system_prompt(patient: dict, session: dict, all_sessions: list | None = None) -> str:
    pname = (patient.get("name") or "Patient").strip()
    age   = patient.get("age", "unknown")
    onset = patient.get("wound_onset_date", "unknown")

    p = (
        f"You are a wound care AI assistant specializing in pressure injury management (NPUAP/EPUAP staging). "
        f"Patient: {pname}, {age} years old, wound onset date: {onset}.\n"
    )

    graded = sorted(
        [s for s in (all_sessions or []) if s.get("grade", -1) >= 0],
        key=lambda s: s.get("date", "")
    )
    if len(graded) > 1:
        p += f"Wound assessment history ({len(graded)} sessions, oldest → latest):\n"
        for s in graded:
            g = int(s.get("grade", 0))
            g_label = _OW_GRADE_INFO[g][1] if 0 <= g <= 5 else "Unknown (idx)"
            area   = s.get("wound_area_mm2") or (s.get("ai_result") or {}).get("detect", {}).get("area_mm2", 0)
            length = s.get("wound_length_mm") or (s.get("ai_result") or {}).get("detect", {}).get("length_mm", 0)
            extras = ""
            if area:   extras += f", area {area} mm²"
            if length: extras += f", length {length} mm"
            marker = " ← latest" if s is graded[-1] else ""
            p += f"  • {s.get('date','?')}: Grade {g} ({g_label}){extras}{marker}\n"

        first_g, last_g = graded[0].get("grade", 0), graded[-1].get("grade", 0)
        if last_g < first_g:
            p += "Trend: Healing (improving).\n"
        elif last_g > first_g:
            p += "Trend: Deteriorating.\n"
        else:
            p += "Trend: Stable.\n"
    else:
        grade = int(session.get("grade", -1))
        p += f"Latest assessment: Grade {grade}"
        if 0 <= grade <= 5:
            p += f" ({_OW_GRADE_INFO[grade][1]}) — {_OW_GRADE_INFO[grade][4]}"
        p += "\n"

    p += "Respond with 2–3 concise bullet points. No preamble."
    return p


def _build_system_prompt_chat(patient: dict, session: dict, all_sessions: list | None = None) -> str:
    pname = (patient.get("name") or "Patient").strip()
    age   = patient.get("age", "unknown")
    onset = patient.get("wound_onset_date", "unknown")

    p = (
        f"You are a helpful wound care AI assistant specializing in pressure injury management (NPUAP/EPUAP staging). "
        f"Patient: {pname}, {age} years old, wound onset date: {onset}.\n"
    )
    graded = sorted(
        [s for s in (all_sessions or []) if s.get("grade", -1) >= 0],
        key=lambda s: s.get("date", "")
    )
    if len(graded) > 1:
        p += f"Wound history ({len(graded)} assessments):\n"
        for s in graded:
            g = int(s.get("grade", 0))
            g_label = _OW_GRADE_INFO[g][1] if 0 <= g <= 5 else "?"
            marker = " ← latest" if s is graded[-1] else ""
            p += f"  • {s.get('date','?')}: Grade {g} ({g_label}){marker}\n"
        first_g, last_g = graded[0]["grade"], graded[-1]["grade"]
        trend = "Healing" if last_g < first_g else "Deteriorating" if last_g > first_g else "Stable"
        p += f"Trend: {trend}.\n"
    elif session:
        grade = int(session.get("grade", -1))
        if grade >= 0:
            p += f"Latest assessment: Grade {grade} ({_OW_GRADE_INFO[grade][1] if 0 <= grade <= 5 else '?'}).\n"
    p += "Answer clearly and helpfully. Adapt length to the question."
    return p


def _call_model(cfg: dict, message: str, system: str, max_tokens: int | None = None) -> tuple[str | None, str | None]:
    import urllib.request as ur
    import urllib.error
    mtype    = cfg.get("type", "")
    endpoint = cfg.get("endpoint", "").rstrip("/")
    if mtype != "api" or not endpoint:
        return None, "Invalid model config"
    payload = json.dumps({
        "message":    message,
        "system":     system,
        "max_tokens": max_tokens if max_tokens is not None else cfg.get("max_tokens", 512),
        "temperature": cfg.get("temperature", 0.7),
        "top_p":      cfg.get("top_p", 0.9),
    }).encode()
    try:
        req = ur.Request(endpoint + "/chat", data=payload,
                         headers={"Content-Type": "application/json"}, method="POST")
        with ur.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
            reply = (data.get("reply") or data.get("response")
                     or data.get("text") or data.get("content"))
            return reply, None
    except urllib.error.URLError as e:
        return None, f"Model server unreachable ({endpoint}) — is it running? ({e.reason})"
    except Exception as e:
        return None, f"Model error: {e}"


def _rule_based_chat(message: str, patient: dict, session: dict, all_sessions: list | None = None) -> str:
    grade     = int(session.get("grade", -1))
    pname     = (patient.get("name") or "the patient").strip()
    pid       = patient.get("patient_id", "—")
    onset     = patient.get("wound_onset_date", "—")
    age       = patient.get("age", "—")
    sess_date = session.get("date", "—")
    msg_lc    = message.lower()

    g_name = g_advice = "—"
    if 0 <= grade <= 5:
        g_name   = _OW_GRADE_INFO[grade][1]
        g_advice = _OW_GRADE_INFO[grade][4]

    graded = sorted(
        [s for s in (all_sessions or []) if s.get("grade", -1) >= 0],
        key=lambda s: s.get("date", "")
    )
    history_lines = ""
    trend_text = ""
    if len(graded) > 1:
        history_lines = "\n".join(
            f"  • {s['date']}: Grade {s['grade']} ({_OW_GRADE_INFO[s['grade']][1] if 0 <= s['grade'] <= 5 else '?'})"
            for s in graded
        )
        first_g, last_g = graded[0]["grade"], graded[-1]["grade"]
        trend_text = "Healing" if last_g < first_g else "Deteriorating" if last_g > first_g else "Stable"

    if any(k in msg_lc for k in ["history", "trend", "progress", "heal", "previous", "all", "past"]):
        if not graded:
            return "No assessments recorded yet."
        resp = f"**{pname}** — {len(graded)} assessment(s):\n{history_lines}"
        if trend_text:
            resp += f"\n**Trend: {trend_text}**"
        return resp

    if any(k in msg_lc for k in ["stage", "grade", "status", "score", "assessment", "wound", "dti", "unstageable"]):
        if grade < 0:
            return "No wound classification recorded yet."
        resp = f"Latest ({sess_date}): **{g_name}**\n- {g_advice}"
        if trend_text:
            resp += f"\n- Overall trend: **{trend_text}** across {len(graded)} visits."
        return resp

    if any(k in msg_lc for k in ["size", "measure", "area", "length", "mm"]):
        det    = (session.get("ai_result") or {}).get("detect", {})
        area   = session.get("wound_area_mm2") or det.get("area_mm2", 0)
        length = session.get("wound_length_mm") or det.get("length_mm", 0)
        if area or length:
            return (
                f"Latest wound measurements for **{pname}**:\n"
                + (f"- Length: **{length} mm**\n" if length else "")
                + (f"- Area: **{area} mm²** (estimated, 10 px/mm assumed)" if area else "")
            )
        return "No AI measurements recorded for the latest session."

    if any(k in msg_lc for k in ["treatment", "recommend", "care", "debride", "advise", "dress", "off-load", "dressing"]):
        if grade < 0:
            return "Please complete a wound classification first."
        resp = f"**{g_name}:** {g_advice}"
        if trend_text:
            resp += f" Trend: {trend_text}."
        return resp

    if any(k in msg_lc for k in ["patient", "name", "id", "age", "onset", "who"]):
        return (
            f"Patient: **{pname}** · Age: **{age}** · ID: **{pid}**\n"
            f"Wound onset: **{onset}**\n"
            + (f"Latest: {sess_date}, Grade {grade} ({g_name})." if grade >= 0 else "No assessment yet.")
            + (f" Trend: {trend_text} ({len(graded)} visits)." if trend_text else "")
        )

    if any(k in msg_lc for k in ["hello", "hi", "hey", "help"]):
        return (
            f"Hello! Open wound assistant for **{pname}**.\n"
            "Ask about: **grade**, **history**, **trend**, **size**, **treatment**, **patient** info."
        )

    if grade >= 0:
        resp = f"**{pname}** — **{g_name}**. {g_advice}"
        if trend_text:
            resp += f" Trend: {trend_text}."
        return resp
    return f"Ready to assist with **{pname}'s** pressure injury monitoring. No classification on record yet."


class DemoChatMsg(BaseModel):
    message: str
    patient: dict = {}
    latest_session: dict = {}
    all_sessions: list = []
    model_id: str = ""
    mode: str = "chat"


@app.post("/api/demo/chat")
def api_demo_chat(body: DemoChatMsg, _: None = Depends(require_auth)):
    used_model  = "rule-based"
    model_error = None
    reply       = None

    if body.model_id:
        adapters = _scan_block3_adapters()
        cfg = adapters.get(_safe(body.model_id))
        if cfg:
            if body.mode == "recommendation":
                system = _build_system_prompt(body.patient, body.latest_session, body.all_sessions)
                reply, err = _call_model(cfg, body.message, system, max_tokens=256)
            else:
                system = _build_system_prompt_chat(body.patient, body.latest_session, body.all_sessions)
                reply, err = _call_model(cfg, body.message, system, max_tokens=1024)
            if reply:
                used_model = cfg.get("name", body.model_id)
            else:
                model_error = err or "Model returned empty response"
        else:
            model_error = f"Model not found: {body.model_id}"

    if not reply:
        if body.mode == "chat":
            if not body.model_id:
                reply = None
                model_error = model_error or "no_model"
        else:
            reply = _rule_based_chat(body.message, body.patient, body.latest_session, body.all_sessions)
            used_model = "rule-based"

    return JSONResponse({"reply": reply, "model": used_model, "model_error": model_error})


# ── AI segmentation ───────────────────────────────────────────────────────────

_detect_predictor = None
_detect_lock = threading.Lock()
_seg_predictor = None
_seg_lock = threading.Lock()


def _get_detect_predictor():
    global _detect_predictor
    if _detect_predictor is None:
        with _detect_lock:
            if _detect_predictor is None:
                dp = str(DETECT_DIR)
                if dp not in sys.path:
                    sys.path.insert(0, dp)
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "detect_wound_infer", DETECT_DIR / "infer.py"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _detect_predictor = mod.WoundDetector(
                    checkpoint_path=DETECT_CHECKPOINT
                )
    return _detect_predictor


def _get_seg_predictor():
    global _seg_predictor
    if _seg_predictor is None:
        with _seg_lock:
            if _seg_predictor is None:
                sp = str(SEG_DIR)
                if sp not in sys.path:
                    sys.path.insert(0, sp)
                from infer import WoundPredictor  # noqa: PLC0415
                _seg_predictor = WoundPredictor(
                    seg_checkpoint=SEG_CHECKPOINT,
                    grade_checkpoint=None,  # grading model not available yet
                )
    return _seg_predictor


_OVERLAY_MAX_DIM = 1600


def _save_overlay_jpg(overlays_dir: Path, name: str, overlay_bgr) -> str:
    """Downscale + JPEG-encode an overlay before saving (raw PNGs from
    full-res photos can be 10+ MB, which stalls loading it in the browser)."""
    import cv2

    h, w = overlay_bgr.shape[:2]
    scale = _OVERLAY_MAX_DIM / max(h, w)
    if scale < 1:
        overlay_bgr = cv2.resize(overlay_bgr, (int(w * scale), int(h * scale)))
    cv2.imwrite(str(overlays_dir / name), overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return name


def _run_segmentation_on_bytes(raw: bytes, filename: str, sess_dir: Path) -> dict:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {"error": "Model dependencies not installed."}

    nparr = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        try:
            import pillow_heif
            from PIL import Image as PILImage
            import io as _io
            pillow_heif.register_heif_opener()
            pil_img = PILImage.open(_io.BytesIO(raw)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return {"error": "Could not decode image (unsupported format)"}

    overlays_dir = sess_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    stem   = Path(filename).stem
    result = {}

    if DETECT_CHECKPOINT.exists():
        try:
            det = _get_detect_predictor()
            d1  = det.predict(img_bgr)
            detect_overlay_name = _save_overlay_jpg(
                overlays_dir, stem + "_detect_overlay.jpg", d1["overlay_bgr"]
            )
            result["detect"] = {
                "overlay_filename": detect_overlay_name,
                "area_px":      d1["area_px"],
                "area_mm2":     d1["area_mm2"],
                "length_mm":    d1.get("length_mm", 0.0),
                "pixel_per_mm": d1["pixel_per_mm"],
            }
        except Exception as exc:
            result["detect"] = {"error": str(exc)}

    predictor = _get_seg_predictor()
    b2 = predictor.predict_with_overlay(img_bgr)
    b2.pop("mask", None)
    overlay_bgr  = b2.pop("overlay_bgr")
    overlay_name = _save_overlay_jpg(overlays_dir, stem + "_overlay.jpg", overlay_bgr)
    b2["overlay_filename"] = overlay_name
    result.update(b2)
    return result


@app.post("/api/demo/session/from_image")
async def api_demo_session_from_image(request: Request, file: UploadFile = File(...)):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    suffix = Path(file.filename or "img.jpg").suffix.lower()
    if suffix not in IMG_EXTS:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    raw = await file.read()
    orig_name = file.filename or ("upload" + suffix)
    if suffix in {".heic", ".heif"}:
        safe_name = _safe(Path(orig_name).stem + ".jpg")
        try:
            import pillow_heif, io as _io
            from PIL import Image as _PILImage
            pillow_heif.register_heif_opener()
            pil_img = _PILImage.open(_io.BytesIO(raw)).convert("RGB")
            buf = _io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=92)
            raw = buf.getvalue()
        except Exception as e:
            return JSONResponse({"error": f"HEIC conversion failed: {e}"}, status_code=400)
    else:
        safe_name = _safe(orig_name)

    with _lock:
        d = _load_ow()
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        sess_dir = OW_IMG_DIR / sid
        sess_dir.mkdir(parents=True, exist_ok=True)
        with open(sess_dir / safe_name, "wb") as fp:
            fp.write(raw)

        ai_result = None
        if SEG_DIR.exists() and SEG_CHECKPOINT.exists():
            try:
                seg = _run_segmentation_on_bytes(raw, safe_name, sess_dir)
                if "error" not in seg:
                    ai_result = {**seg, "source_image": safe_name}
            except Exception as exc:
                ai_result = {"error": str(exc), "source_image": safe_name}

        grade = 0
        if ai_result and "grade" in ai_result and "error" not in ai_result:
            grade = max(0, min(5, ai_result["grade"]))

        wound_area   = 0.0
        wound_length = 0.0
        if ai_result and "detect" in ai_result and "error" not in ai_result.get("detect", {}):
            wound_area   = round(ai_result["detect"].get("area_mm2",  0.0), 2)
            wound_length = round(ai_result["detect"].get("length_mm", 0.0), 2)

        sess = {
            "id": sid,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "grade": grade,
            "wound_width": 0.0,
            "wound_height": 0.0,
            "wound_area_mm2":  wound_area,
            "wound_length_mm": wound_length,
            "notes": "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "images": [safe_name],
            "videos": [],
            "ai_result": ai_result,
        }
        d.setdefault("sessions", []).append(sess)
        _save_ow(d)

    return JSONResponse({"ok": True, "session": sess, "data": _load_ow()})


@app.post("/api/demo/session/from_video")
async def api_demo_session_from_video(request: Request, file: UploadFile = File(...)):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_EXTS:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    raw      = await file.read()
    safe_name = _safe(file.filename or ("video" + suffix))

    with _lock:
        d = _load_ow()
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        sess_dir = OW_IMG_DIR / sid
        sess_dir.mkdir(parents=True, exist_ok=True)
        with open(sess_dir / safe_name, "wb") as fp:
            fp.write(raw)

        sess = {
            "id": sid,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "grade": -1,
            "wound_width": 0.0,
            "wound_height": 0.0,
            "wound_area_mm2": 0.0,
            "notes": "Video uploaded — AI analysis pending.",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "images": [],
            "videos": [safe_name],
            "ai_result": None,
        }
        d.setdefault("sessions", []).append(sess)
        _save_ow(d)

    return JSONResponse({"ok": True, "session": sess, "data": _load_ow()})


@app.get("/api/demo/session/{sid}/segment/{filename}")
def api_demo_session_segment(sid: str, filename: str, request: Request):
    if not is_authed(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    path = OW_IMG_DIR / _safe(sid) / _safe(filename)
    if not path.exists():
        return JSONResponse({"error": "Image not found"}, status_code=404)

    if not SEG_DIR.exists():
        return JSONResponse({"error": "Segmentation model not found."}, status_code=503)

    try:
        import base64
        import cv2
        import numpy as np
        predictor = _get_seg_predictor()
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)
        result = predictor.predict_with_overlay(img_bgr)
        overlay_png = cv2.imencode(".png", result.pop("overlay_bgr"))[1].tobytes()
        result["overlay_png_base64"] = base64.b64encode(overlay_png).decode("ascii")
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/demo/segment/status")
def api_seg_status(_: None = Depends(require_auth)):
    return JSONResponse({
        "loaded": _seg_predictor is not None,
        "available": SEG_DIR.exists() and SEG_CHECKPOINT.exists(),
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print(">>> Open Wound Monitor — server is running.")
    print(">>> Open in your browser: http://localhost:8001")
    print(f">>> Login -> username: {USERNAME}   password: {PASSWORD}")
    print("=" * 60)
    print(f">>> Data folder:  {OW_IMG_DIR}")
    print(f">>> AI models:    {AI_MODEL_DIR}")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001)
