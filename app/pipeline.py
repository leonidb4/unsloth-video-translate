"""Оркестрация дубляжа: шаги, статусы job, чекпоинты для перезапуска с места ошибки."""

from __future__ import annotations

import json
import os
import shutil
import time

import httpx

from . import ffmpeg_utils as ff
from .translate import translate_en_ru
from .unsloth_client import UnslothClient

STEPS = [
    "Извлечение аудиодорожки",
    "Разделение голоса и музыки",
    "Транскрибация с определением говорящих",
    "Перевод на русский",
    "Клонирование голосов",
    "Синтез речи",
    "Сборка итогового видео",
]

VOCAL_HINTS = ("vocal", "voice", "speech")
MUSIC_HINTS = ("instrumental", "music", "accompaniment", "drums", "bass")
SPEAKER_KEYS = ("speaker", "speaker_id", "spk", "channel")


def _set_step(job: dict, idx: int) -> None:
    if idx > job["step"]:
        job["detail"] = None  # завершаем шаг — его описание не несём дальше
    now = time.time()
    for i, st in enumerate(job["steps"]):
        if i == idx and st["state"] != "active":
            st["started"] = now  # запуск таймера шага (и сброс при повторном входе)
            st["elapsed"] = None
        elif i < idx and st.get("elapsed") is None and st.get("started"):
            st["elapsed"] = now - st["started"]  # шаг завершён — фиксируем затраченное время
        st["state"] = "done" if i < idx else ("active" if i == idx else "pending")
    job["step"] = idx


def _log(job: dict, msg: str) -> None:
    job["log"].append(msg)
    del job["log"][:-40]


def _detail(job: dict, msg: str) -> None:
    """Текущее описание активного шага для UI — одна живая строка."""
    job["detail"] = msg


class JobCancelled(Exception):
    """Задача отменена пользователем (не ошибка)."""


def _check_cancel(job: dict) -> None:
    """Бросить JobCancelled, если нажата «Отменить». Вызывается между подшагами."""
    ev = job.get("cancel")
    if ev is not None and ev.is_set():
        raise JobCancelled()


def _settle_active_step(job: dict) -> None:
    """Зафиксировать время активного шага, когда задача завершилась (error/canceled):
    иначе в UI таймер живого шага продолжает тикать."""
    now = time.time()
    for s in job["steps"]:
        if s["state"] == "active" and s.get("started"):
            s["elapsed"] = now - s["started"]


# -- чекпоинты: что уже сделано, чтобы retry не повторял проделанное -----------------

def _save_state(st: dict, workdir: str) -> None:
    with open(os.path.join(workdir, "state.json"), "w") as f:
        json.dump(st, f, ensure_ascii=False)


def _persist_elapsed(st: dict, job: dict) -> None:
    # время шагов сохраняем в state.json — после retry пропущенные шаги
    # показывают затраченное время из прошлой попытки
    st["elapsed"] = {str(i): s["elapsed"] for i, s in enumerate(job["steps"]) if s.get("elapsed") is not None}


def _checkpoint(st: dict, workdir: str, step: int, job: dict) -> None:
    st["completed_step"] = step
    _persist_elapsed(st, job)
    _save_state(st, workdir)


def _finish_state(st: dict, workdir: str, status: str, error: str | None = None) -> None:
    """Финальный статус в state.json — чтобы после рестарта приложения задача
    восстановилась с правильным статусом, а не перезапустилась заново."""
    st["status"] = status
    if error is not None:
        st["error"] = error
    _save_state(st, workdir)


def _load_state(workdir: str) -> dict:
    try:
        with open(os.path.join(workdir, "state.json")) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _cleanup_intermediates(workdir: str) -> None:
    """Удалить промежуточные дорожки и сегменты после успешной сборки: итоговое
    видео уже перенесено в storage. Исходник (upload.*) и state.json сохраняем —
    без них задача не восстановится после рестарта приложения."""
    for name in os.listdir(workdir):
        if name.lower().endswith((".wav", ".mp3")):
            try:
                os.remove(os.path.join(workdir, name))
            except OSError:  # best-effort: сбой уборки не портит готовую задачу
                pass


def _parse_segments(data: dict) -> list[dict]:
    """Сегменты транскрипции: текст + таймкоды + спикер (defensive-парсинг)."""
    raw = list(data.get("segments") or [])
    if not raw:  # формат OpenAI verbose: сегменты могут лежать в channels[]
        for ch in data.get("channels") or []:
            for s in ch.get("segments") or []:
                s = dict(s)
                s.setdefault("channel", ch.get("id"))
                raw.append(s)

    segs: list[dict] = []
    for s in raw:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = None
        for k in SPEAKER_KEYS:
            v = s.get(k)
            if v is not None:
                spk = str(v)
                break
        segs.append(
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": text,
                "speaker": spk,
            }
        )
    if segs and all(x["speaker"] is None for x in segs):
        for x in segs:
            x["speaker"] = "0"
    return segs


def _clip_label(clip: dict) -> str:
    return " ".join(str(clip.get(k) or "") for k in ("name", "type", "label", "kind")).lower()


def _pick_stems(clips: list[dict], client: UnslothClient, workdir: str, job: dict) -> tuple[str, str | None]:
    """Скачать дорожки и определить, какая vocals, а какая instrumental."""
    paths: dict[int, str] = {}
    for i, c in enumerate(clips):
        _check_cancel(job)
        _detail(job, f"Скачиваю дорожку {i + 1}/{len(clips)}…")
        p = os.path.join(workdir, f"stem_{i}.wav")
        client.download_clip(c, p)
        paths[i] = p

    if len(clips) == 1:
        return paths[0], None

    vocal_i = music_i = None
    for i, c in enumerate(clips):
        lab = _clip_label(c)
        if any(h in lab for h in VOCAL_HINTS):
            vocal_i = i
        elif any(h in lab for h in MUSIC_HINTS):
            music_i = i

    if vocal_i is None or music_i is None:
        _log(job, "Имена дорожек неизвестны — определяю голос по транскрипции")
        best, best_len = None, -1
        for i, p in paths.items():
            _detail(job, "Определяю голосовую дорожку по транскрипции…")
            n = len((client.transcribe(p).get("text") or "").strip())
            if n > best_len:
                best, best_len = i, n
        vocal_i = best
        music_i = next(i for i in paths if i != vocal_i)

    return paths[vocal_i], (paths[music_i] if music_i is not None else None)


def run_job(job: dict, workdir: str, upload_path: str, storage_dir: str) -> None:
    """Полный pipeline. Работает в отдельном потоке; результаты пишет в job.

    Шаги с сохранённым чекпоинтом (state.json в workdir) пропускаются —
    так retry после ошибки продолжает с места сбоя, а не заново.
    """
    client = UnslothClient()
    job["client"] = client  # чтобы endpoint отмены мог оборвать идущий запрос
    st = _load_state(workdir)
    done = st.get("completed_step", -1)
    for i, s in enumerate(job["steps"]):  # время пропущенных (checkpoint) шагов из прошлой попытки
        e = st.get("elapsed", {}).get(str(i))
        if e is not None and i <= done:
            s["elapsed"] = e
    try:
        audio_wav = os.path.join(workdir, "audio.wav")

        # 0. Аудио из видео
        if done < 0:
            _check_cancel(job)
            _set_step(job, 0)
            _detail(job, "В процессе…")
            ff.extract_audio(upload_path, audio_wav)
            _checkpoint(st, workdir, 0, job)

        # 1. Разделение голоса/музыки
        if done < 1:
            _check_cancel(job)
            _set_step(job, 1)
            _detail(job, "Загружаю аудио на API…")
            src_id = client.upload_input(audio_wav, "audio.wav")
            st["src_id"] = src_id
            _detail(job, "В процессе…")
            clips = client.separate(src_id)
            vocals_wav, music_wav = _pick_stems(clips, client, workdir, job)
            st["vocals"] = os.path.basename(vocals_wav)
            st["music"] = os.path.basename(music_wav) if music_wav else None
            _checkpoint(st, workdir, 1, job)
        else:
            vocals_wav = os.path.join(workdir, st["vocals"])
            music_wav = os.path.join(workdir, st["music"]) if st.get("music") else None

        # 2. Транскрибация с диаризацией
        if done < 2:
            _check_cancel(job)
            _set_step(job, 2)
            _detail(job, "В процессе…")
            segments = _parse_segments(client.transcribe(vocals_wav))
            if not segments:
                raise RuntimeError("Речь в видео не обнаружена")
            st["segments"] = segments
            _checkpoint(st, workdir, 2, job)
            n_speakers = len({s["speaker"] for s in segments})
            _log(job, f"Сегментов: {len(segments)}, говорящих: {n_speakers}")
        else:
            segments = st["segments"]

        # 3. Перевод en → ru
        if done < 3:
            _check_cancel(job)
            _set_step(job, 3)

            def _report_batch(d: int, total: int) -> None:
                _detail(job, f"Перевожу на русский (партия {d}/{total})…")
                _check_cancel(job)  # отмена между партиями перевода

            ru_texts = translate_en_ru([s["text"] for s in segments], client, on_batch=_report_batch)
            for seg, ru in zip(segments, ru_texts):
                seg["ru"] = ru
            st["segments"] = segments  # теперь с "ru"
            _checkpoint(st, workdir, 3, job)

        # 4. Клонирование голосов (при retry — только ещё не созданные)
        if done < 4:
            _check_cancel(job)
            _set_step(job, 4)
            voice_ids: dict[str, str] = st.get("voice_ids", {})
            speakers = sorted({s["speaker"] for s in segments}, key=lambda x: (len(x), x))
            for n, spk in enumerate(speakers, 1):
                _check_cancel(job)
                if spk in voice_ids:
                    _detail(job, f"Голос спикера {spk} уже есть — пропускаю")
                    continue
                _detail(job, f"Клонирую голос спикера {spk} ({n}/{len(speakers)})…")
                spk_segs = [s for s in segments if s["speaker"] == spk]
                # Запись и транскрипт должны совпадать (модель клонирует то, что в записи):
                # берём префикс фрагментов спикера до ~9 секунд — те же, что уйдут в обе части
                chosen: list[dict] = []
                acc = 0.0
                for s in spk_segs:
                    d = s["end"] - s["start"]
                    if chosen and acc + d > 9.0:
                        break
                    chosen.append(s)
                    acc += d
                sample = ff.cut_concat(
                    vocals_wav, [(s["start"], s["end"]) for s in chosen], os.path.join(workdir, f"sample_{n}.wav")
                )
                transcript = " ".join(s["text"] for s in chosen)
                if len(chosen) == 1 and (chosen[0]["end"] - chosen[0]["start"]) > 9.0:
                    # единственный длинный фрагмент cut_concat всё же обрежет до 9 с — текст укорачиваем пропорционально
                    words = chosen[0]["text"].split()
                    transcript = " ".join(words[: max(1, int(len(words) * 9.0 / (chosen[0]["end"] - chosen[0]["start"])))])
                transcript = transcript[:2000]
                if sample is None or not transcript:  # спикер без уловимой речи — общий сэмпл
                    _log(job, f"Спикер {spk}: нет собственных фрагментов, использую начало видео")
                    sample = ff.cut_concat(
                        vocals_wav, [(segments[0]["start"], segments[0]["end"])], os.path.join(workdir, f"sample_{n}.wav")
                    )
                    transcript = segments[0]["text"][:2000]
                if sample is None:
                    raise RuntimeError(f"Не удалось вырезать голосовой сэмпл для спикера {spk}")
                rec_id = client.upload_input(sample, f"voice_{n}.wav")
                voice_ids[spk] = client.create_voice(f"Speaker {n}", rec_id, transcript)
                st["voice_ids"] = voice_ids
                _save_state(st, workdir)  # чтобы retry не клонировал заново
                _log(job, f"Голос спикера {spk} готов (голос #{n})")
            _checkpoint(st, workdir, 4, job)

        # 5. TTS каждого сегмента голосом его спикера (при retry — только недостающие файлы)
        if done < 5:
            _check_cancel(job)
            _set_step(job, 5)
            voice_ids = st.get("voice_ids", {})
            for i, seg in enumerate(segments):
                _check_cancel(job)
                out = os.path.join(workdir, f"seg_{i:03d}.mp3")
                if os.path.exists(out) and os.path.getsize(out) > 0:
                    continue
                _detail(job, f"Озвучиваю сегмент {i + 1}/{len(segments)}…")
                client.speech(voice_ids[seg["speaker"]], seg["ru"], out)
            _checkpoint(st, workdir, 5, job)

        # 6. Сборка: подгонка по таймкодам + микс с музыкой + mux
        if done < 6:
            _check_cancel(job)
            _set_step(job, 6)
            total = ff.probe_duration(upload_path)
            timeline: list[tuple[str, float]] = []
            for i, seg in enumerate(segments):
                _check_cancel(job)
                _detail(job, f"Подгоняю сегмент {i + 1}/{len(segments)} по таймкодам…")
                target = max(0.5, seg["end"] - seg["start"])
                fit_wav = os.path.join(workdir, f"seg_{i:03d}_fit.wav")
                ff.fit_duration(os.path.join(workdir, f"seg_{i:03d}.mp3"), fit_wav, target)
                timeline.append((fit_wav, seg["start"]))
            _check_cancel(job)
            _detail(job, "Микширую речь с музыкальной дорожкой…")
            mixed_wav = os.path.join(workdir, "mixed.wav")
            ff.mix_timeline(timeline, music_wav, mixed_wav, total)
            _check_cancel(job)
            _detail(job, "В процессе…")
            out_mp4 = os.path.join(workdir, "final.mp4")
            ff.mux_video(upload_path, mixed_wav, out_mp4)

            final_path = os.path.join(storage_dir, f"{job['id']}.mp4")
            shutil.move(out_mp4, final_path)
            job["video_path"] = final_path
            _checkpoint(st, workdir, 6, job)

        _set_step(job, len(STEPS))
        job["status"] = "done"
        _finish_state(st, workdir, "done")
        _cleanup_intermediates(workdir)
    except JobCancelled:  # отмена по проверке между подшагами — не ошибка, свой статус
        job["status"] = "canceled"
        _settle_active_step(job)
        _persist_elapsed(st, job)
        _finish_state(st, workdir, "canceled")
    except Exception as e:  # noqa: BLE001 — любой сбой шага показываем в UI
        _settle_active_step(job)
        _persist_elapsed(st, job)
        ev = job.get("cancel")
        if ev is not None and ev.is_set():
            # соединение оборвано отменой (close клиента) — тоже не ошибка
            job["status"] = "canceled"
        elif isinstance(e, httpx.TimeoutException):  # таймаут API — отдельное понятное сообщение
            job["status"] = "error"
            job["error"] = f"API не ответил за отведённое время ({type(e).__name__}) — операция слишком долгая, нажмите «Повторить с места ошибки»"
        else:
            job["status"] = "error"
            job["error"] = str(e) or e.__class__.__name__
        _finish_state(st, workdir, job["status"], job.get("error"))
    finally:
        client.close()
