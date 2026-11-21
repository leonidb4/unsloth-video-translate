"""Web-приложение: дубляж видео en → ru через Unsloth API."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import STORAGE_DIR, WORK_DIR as WORK_ROOT, api_settings_error
from .pipeline import STEPS, _cleanup_intermediates, _load_state, run_job

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# WORK_ROOT — каталог рабочих файлов (чекпоинты, дорожки, сегменты), задаётся в .env
# (WORK_DIR, дефолт ./work рядом с кодом). Постоянный: в /tmp после перезагрузки
# ничего не остаётся и прогресс задачи теряется.

JOBS: dict[str, dict] = {}


LEGACY_WORK_ROOT = "/tmp/unsloth-jobs"  # где лежали рабочие файлы до переезда в WORK_ROOT


def _restore_jobs() -> None:
    """Восстановить задачи после рестарта приложения.

    Рабочие файлы живут в WORK_ROOT (state.json — чекпоинты), поэтому:
    готовая задача возвращается как done, оборванная посреди работы —
    перезапускается с места ошибки, error/canceled — показываются со своим статусом.
    Остатки из старого /tmp/unsloth-jobs переезжают в WORK_ROOT.
    """
    if os.path.isdir(LEGACY_WORK_ROOT):
        for job_id in os.listdir(LEGACY_WORK_ROOT):
            src = os.path.join(LEGACY_WORK_ROOT, job_id)
            if os.path.isdir(src) and not os.path.exists(os.path.join(WORK_ROOT, job_id)):
                shutil.move(src, os.path.join(WORK_ROOT, job_id))
    if not os.path.isdir(WORK_ROOT):
        return
    for job_id in sorted(os.listdir(WORK_ROOT)):
        workdir = os.path.join(WORK_ROOT, job_id)
        if not os.path.isdir(workdir) or job_id in JOBS:
            continue
        upload = next(
            (os.path.join(workdir, f) for f in os.listdir(workdir) if f.startswith("upload")), None
        )
        if upload is None:
            continue  # без исходного файла задачу восстановить нельзя
        st = _load_state(workdir)
        done_step = st.get("completed_step", -1)
        steps = [{"name": n, "state": "pending", "started": None, "elapsed": None} for n in STEPS]
        for i, s in enumerate(steps):
            if i <= done_step:
                s["state"] = "done"
            e = st.get("elapsed", {}).get(str(i))
            if e is not None and i <= done_step:
                s["elapsed"] = e
        video = os.path.join(STORAGE_DIR, f"{job_id}.mp4")
        job = {
            "id": job_id,
            "status": "processing",
            "step": min(max(done_step + 1, 0), len(STEPS)),
            "steps": steps,
            "detail": None,
            "error": st.get("error"),
            "video_path": video if os.path.exists(video) else None,
            "log": [],
            "created": os.path.getmtime(upload),
            "cancel": threading.Event(),
        }
        if os.path.exists(video):
            job["status"] = "done"
            for s in steps:
                s["state"] = "done"
            _cleanup_intermediates(workdir)  # приложение могло умереть между «done» и уборкой
        elif st.get("status") in ("error", "canceled"):
            job["status"] = st["status"]
        else:
            # приложение умерло во время работы — продолжаем с чекпоинта
            threading.Thread(target=run_job, args=(job, workdir, upload, STORAGE_DIR), daemon=True).start()
        JOBS[job_id] = job


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not STORAGE_DIR:
        raise RuntimeError("STORAGE_DIR не задан — укажите путь к каталогу готовых видео в .env")
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(WORK_ROOT, exist_ok=True)
    _restore_jobs()
    err = api_settings_error()
    if err:
        print(f"[warn] {err} — обработка будет недоступна, пока не заполнен .env", flush=True)
    yield


app = FastAPI(title="Unsloth Video Translate", lifespan=lifespan)


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    err = api_settings_error()
    if err:
        raise HTTPException(status_code=503, detail=err)

    # ID задачи — SHA-256 содержимого видео (первые 12 hex): то же видео
    # всегда даёт один и тот же id. Повторная загрузка готовой/идущей задачи
    # возвращает его; после ошибки или отмены — перезапускает с чекпоинта.
    hasher = hashlib.sha256()
    file.file.seek(0)
    while chunk := file.file.read(1024 * 1024):
        hasher.update(chunk)
    job_id = hasher.hexdigest()[:12]

    existing = JOBS.get(job_id)
    if existing and existing["status"] not in ("error", "canceled"):
        return {"id": job_id}

    workdir = os.path.join(WORK_ROOT, job_id)
    os.makedirs(workdir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    upload_path = os.path.join(workdir, f"upload{ext}")
    file.file.seek(0)
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job: dict = {
        "id": job_id,
        "status": "processing",
        "step": 0,
        "steps": [
            {"name": name, "state": "pending", "started": None, "elapsed": None} for name in STEPS
        ],
        "detail": None,
        "error": None,
        "video_path": None,
        "log": [],
        "created": time.time(),
        "cancel": threading.Event(),  # set() = «стоп на следующей проверке»
    }
    JOBS[job_id] = job
    threading.Thread(target=run_job, args=(job, workdir, upload_path, STORAGE_DIR), daemon=True).start()
    return {"id": job_id}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Отменить задачу. Идущий HTTP-запрос обрывается сразу (аналог AbortSignal):
    close() клиента рвёт соединение, и запрос падает с ошибкой в потоке pipeline.
    Если запрос не идёт — поток остановится на следующей проверке."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if job["status"] != "processing":
        raise HTTPException(status_code=409, detail="Отменить можно только задачу в обработке")
    job["cancel"].set()  # сначала флаг: поток по нему поймёт, что сбой — отмена, а не ошибка
    client = job.get("client")
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001 — клиент мог успеть закрыться сам
            pass
    job["status"] = "cancelling"  # UI сразу отреагирует; финальный статус поставит поток
    return {"id": job_id, "status": job["status"]}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    """Повторить задачу с места ошибки/отмены: чекпоинты в workdir сохраняют проделанное,
    исходный файл уже лежит там — повторять загрузку не нужно."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if job["status"] not in ("error", "canceled"):
        raise HTTPException(status_code=409, detail="Повторить можно только задачу с ошибкой или отменённую")
    workdir = os.path.join(WORK_ROOT, job_id)
    upload = next((os.path.join(workdir, f) for f in os.listdir(workdir) if f.startswith("upload")), None)
    if upload is None:
        raise HTTPException(status_code=409, detail="Исходный файл не найден — загрузите видео заново")

    st = _load_state(workdir)
    done = st.get("completed_step", -1)
    for i, s in enumerate(job["steps"]):  # сбросить шаги после чекпоинта (включая упавший «active»)
        if i > done:
            s.update(state="pending", started=None, elapsed=None)
    job["status"] = "processing"
    job["error"] = None
    job["detail"] = None
    job["cancel"].clear()  # иначе новый поток сразу сочтёт задачу отменённой
    threading.Thread(target=run_job, args=(job, workdir, upload, STORAGE_DIR), daemon=True).start()
    return {"id": job_id, "status": job["status"]}


def _summary(job: dict) -> dict:
    return {
        "id": job["id"],
        "status": job["status"],
        "step": job["step"],
        "detail": job.get("detail"),
        "error": job["error"],
        "video": f"/api/jobs/{job['id']}/video" if job["status"] == "done" else None,
        "created": job.get("created", 0),
    }


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    """Список всех задач (новые сверху) — чтобы вернуться к прогрессу после обновления страницы."""
    return [_summary(j) for j in sorted(JOBS.values(), key=lambda j: j.get("created", 0), reverse=True)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return {**_summary(job), "steps": job["steps"], "log": job["log"][-30:]}


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str, download: bool = False) -> FileResponse:
    job = JOBS.get(job_id)
    if not job or job["status"] != "done" or not job["video_path"]:
        raise HTTPException(status_code=404, detail="Видео ещё не готово")
    return FileResponse(
        job["video_path"],
        media_type="video/mp4",
        filename=f"dubbed_{job_id}.mp4" if download else None,
    )


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
