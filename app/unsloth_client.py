"""Тонкий клиент Unsloth API (см. примеры в README.md)."""

from __future__ import annotations

import json
import os

import httpx

from .config import UNSLOTH_API_BASE, UNSLOTH_API_KEY, UNSLOTH_TRANSLATE_MODEL

SEPARATE_MODEL = "audio-cpp/audio.cpp-gguf/BS-RoFormer-ep368-GGUF"
TRANSCRIBE_MODEL = "audio-cpp/audio.cpp-gguf/MOSS-Transcribe-Diarize-GGUF"
TTS_MODEL = "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-Base-GGUF"

_TRANSLATE_SYSTEM_PROMPT = (
    "Ты — переводчик дубляжа с английского на русский. Переводи реплики естественно, в разговорном стиле, "
    "сохраняя смысл и близкую к оригиналу длину (это важно для тайминга озвучки). "
    "Не добавляй пояснений и комментариев."
)


class UnslothError(RuntimeError):
    pass


class UnslothClient:
    def __init__(self, base_url: str = UNSLOTH_API_BASE, api_key: str = UNSLOTH_API_KEY, timeout: float = 3600.0):
        # read-таймаут длинный: /v1/audio/run и speech — синхронные тяжёлые вызовы;
        # connect короткий: недоступный API падать быстро, а не ждать час
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=15.0),
        )

    def close(self) -> None:
        self._http.close()

    def _check(self, resp: httpx.Response, what: str) -> dict:
        if resp.status_code >= 400:
            raise UnslothError(f"{what}: HTTP {resp.status_code} — {resp.text[:300]}")
        return resp.json()

    def upload_input(self, path: str, name: str) -> str:
        """Загрузить аудиофайл как input, вернуть его id.

        Как в README: `curl --data-binary @file` — сырой бинарный body, без multipart.
        """
        with open(path, "rb") as f:
            resp = self._http.post(
                "/v1/audio/inputs",
                params={"name": name},
                content=f.read(),
                # curl --data-binary по умолчанию шлёт именно этот Content-Type
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        data = self._check(resp, "upload_input")
        input_id = data.get("id")
        if not input_id:
            raise UnslothError(f"upload_input: в ответе нет id: {data}")
        return input_id

    def separate(self, source_input_id: str) -> list[dict]:
        """Разделить аудио на дорожки. Возвращает список clips (id, url, ...)."""
        payload = {
            "workflow": "separate",
            "model": SEPARATE_MODEL,
            "inputs": {"source": {"input_id": source_input_id}},
        }
        data = self._check(self._http.post("/v1/audio/run", json=payload), "separate")
        clips = data.get("clips") or []
        if not clips:
            raise UnslothError(f"separate: в ответе нет clips: {data}")
        return clips

    def download_clip(self, clip: dict, out_path: str) -> None:
        url = clip.get("url")
        if not url:
            raise UnslothError(f"download_clip: в клипе нет url: {clip}")
        resp = self._http.get(url)
        if resp.status_code >= 400:
            raise UnslothError(f"download_clip: HTTP {resp.status_code}")
        with open(out_path, "wb") as f:
            f.write(resp.content)

    def transcribe(self, path: str, language: str = "en") -> dict:
        """Транскрибация с диаризацией. Возвращает verbose_json."""
        name = os.path.basename(path)
        with open(path, "rb") as f:
            resp = self._http.post(
                "/v1/audio/transcriptions",
                data={
                    "model": TRANSCRIBE_MODEL,
                    "response_format": "verbose_json",
                    "language": language,
                    "timestamp_granularities[]": "segment",
                },
                files={"file": (name, f)},
            )
        return self._check(resp, "transcribe")

    def create_voice(self, name: str, recording_input_id: str, transcript: str) -> str:
        payload = {
            "name": name,
            "source": {"input_id": recording_input_id},
            "transcript": transcript,
        }
        data = self._check(self._http.post("/v1/audio/voices", json=payload), "create_voice")
        voice_id = data.get("id")
        if not voice_id:
            raise UnslothError(f"create_voice: в ответе нет id: {data}")
        return voice_id

    def speech(self, voice_id: str, text: str, out_path: str) -> None:
        payload = {"model": TTS_MODEL, "input": text, "voice": voice_id, "response_format": "mp3"}
        resp = self._http.post("/v1/audio/speech", json=payload)
        if resp.status_code >= 400:
            raise UnslothError(f"speech: HTTP {resp.status_code} — {resp.text[:300]}")
        with open(out_path, "wb") as f:
            f.write(resp.content)

    def translate_texts(self, texts: list[str]) -> list[str]:
        """Перевести партии реплик en → ru своим LLM (OpenAI-совместимый chat).
        Ответ ждём строго JSON-массивом строк той же длины.
        """
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
        user_prompt = (
            f"Переведи с английского на русский {len(texts)} реплик ниже. "
            "Ответь ТОЛЬКО JSON-массивом из строк в том же порядке, без нумерации и пояснений: "
            "[\"перевод 1\", \"перевод 2\", ...]\n\n" + numbered
        )
        payload: dict = {
            "messages": [
                {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "stream": False,
        }
        if UNSLOTH_TRANSLATE_MODEL:
            # Явное имя модели — иначе «модель по умолчанию» может оказаться не-LLM (см. config.py)
            payload["model"] = UNSLOTH_TRANSLATE_MODEL
        data = self._check(self._http.post("/v1/chat/completions", json=payload), "translate_texts")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise UnslothError(f"translate_texts: неожиданный формат ответа: {str(data)[:300]}") from e

        # defensive-парсинг: модель иногда оборачивает JSON в markdown-блок
        start, end = content.find("["), content.rfind("]")
        if start == -1 or end <= start:
            raise UnslothError(f"translate_texts: в ответе не найден JSON-массив: {content[:300]!r}")
        try:
            out = json.loads(content[start : end + 1])
        except json.JSONDecodeError as e:
            raise UnslothError(f"translate_texts: битый JSON в ответе: {e}; {content[:300]!r}") from e
        if not isinstance(out, list) or len(out) != len(texts) or any(not isinstance(x, str) for x in out):
            raise UnslothError(
                f"translate_texts: ожидался массив из {len(texts)} строк, получил {type(out).__name__} длины "
                f"{len(out) if isinstance(out, list) else '?'}"
            )
        return [x.strip() for x in out]
