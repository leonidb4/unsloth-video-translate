"""Обёртки над ffmpeg/ffprobe для нужд pipeline."""

from __future__ import annotations

import json
import subprocess


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg: {' '.join(cmd[:8])}…\n{p.stderr[-1500:]}")


def probe_duration(path: str) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe: {p.stderr[-500:]}")
    return float(json.loads(p.stdout)["format"]["duration"])


def extract_audio(video: str, out_wav: str) -> None:
    """Вытащить моно-аудио 44.1 кГц из видео."""
    _run(["ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "44100", out_wav])


def cut_concat(source: str, spans: list[tuple[float, float]], out_wav: str, max_seconds: float = 9.0) -> str | None:
    """Склеить фрагменты source (spans в хронологическом порядке) до max_seconds."""
    selected: list[tuple[float, float]] = []
    total = 0.0
    for start, end in spans:
        if total >= max_seconds:
            break
        take = min(end - start, max_seconds - total)
        if take <= 0.1:
            continue
        selected.append((start, start + take))
        total += take
    if not selected:
        return None

    cmd = ["ffmpeg", "-y"]
    for _ in selected:
        cmd += ["-i", source]
    parts = [
        f"[{i}:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[v{i}]"
        for i, (s, e) in enumerate(selected)
    ]
    parts.append("".join(f"[v{i}]" for i in range(len(selected))) + f"concat=n={len(selected)}:v=0:a=1[out]")
    _run(cmd + ["-filter_complex", ";".join(parts), "-map", "[out]", "-ac", "1", "-ar", "44100", out_wav])
    return out_wav


def fit_duration(src: str, dst: str, target: float) -> None:
    """Подогнать длительность src под target через atempo (0.4–3.0), затем apad/atrim."""
    if target <= 0:
        target = probe_duration(src)
    dur = probe_duration(src)
    ratio = max(0.4, min(3.0, dur / target))

    tempos: list[str] = []
    r = ratio
    while r > 2.0:
        tempos.append("2.0")
        r /= 2.0
    while r < 0.5:
        tempos.append("0.5")
        r /= 0.5  # произведение цепочки должно остаться = ratio; atempo не ниже 0.5
    tempos.append(f"{r:.4f}")

    af = ",".join(f"atempo={t}" for t in tempos) + f",apad,atrim=duration={target:.3f},asetpts=PTS-STARTPTS"
    _run(["ffmpeg", "-y", "-i", src, "-af", af, "-ac", "1", "-ar", "44100", dst])


def mix_timeline(clips: list[tuple[str, float]], music_wav: str | None, out_wav: str, total: float) -> None:
    """Разложить озвученные реплики (path, start_sec) на таймлайне поверх музыки."""
    cmd = ["ffmpeg", "-y"]
    parts: list[str] = []
    mix_labels: list[str] = []

    idx = 0
    if music_wav:
        cmd += ["-i", music_wav]
        parts.append(f"[{idx}:a]atrim=duration={total:.3f},asetpts=PTS-STARTPTS[m]")
        mix_labels.append("[m]")
        idx += 1

    for i, (path, start) in enumerate(clips):
        cmd += ["-i", path]
        ms = int(max(0.0, start) * 1000)
        parts.append(f"[{idx}:a]adelay={ms}|{ms}[c{i}]")
        mix_labels.append(f"[c{i}]")
        idx += 1

    if not clips:
        raise RuntimeError("mix_timeline: нет реплик для смикширования")

    # apad/atrim: без музыкальной дорожки микс закончится на последней реплике,
    # и -shortest в mux_video обрежет хвост видео — доводим аудио до полной длительности
    parts.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:normalize=0,apad,atrim=duration={total:.3f},asetpts=PTS-STARTPTS[out]"
    )
    _run(cmd + ["-filter_complex", ";".join(parts), "-map", "[out]", "-ac", "2", "-ar", "44100", out_wav])


def mux_video(video: str, audio_wav: str, out_mp4: str) -> None:
    """Заменить звуковую дорожку видео, видео-поток не перекодировать."""
    _run(
        [
            "ffmpeg", "-y", "-i", video, "-i", audio_wav,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_mp4,
        ]
    )
