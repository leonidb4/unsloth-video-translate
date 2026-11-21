"""Перевод en → ru LLM, загруженной на Unsloth API (см. UnslothClient.translate_texts).

Google Translate больше не используется: сторонние запросы = риск троттлинга/бана IP.
"""

from __future__ import annotations

from collections.abc import Callable

from .unsloth_client import UnslothClient

BATCH_SIZE = 15


def translate_en_ru(texts: list[str], client: UnslothClient, on_batch: Callable[[int, int], None] | None = None) -> list[str]:
    """Перевести список текстов партиями, сохраняя порядок (пустые строки дают пустой результат).

    on_batch(done_batches, total_batches) вызывается после каждой партии.
    """
    results: list[str] = []
    total = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for done, i in enumerate(range(0, len(texts), BATCH_SIZE), 1):
        batch = texts[i : i + BATCH_SIZE]
        out = client.translate_texts(batch)
        if len(out) != len(batch):
            # иначе сегменты без "ru" будут зачекпоинчены, а на шаге TTS упадёт KeyError
            raise RuntimeError(f"Перевод: ожидалось {len(batch)} результатов, получено {len(out)}")
        results.extend((t or "").strip() for t in out)
        if on_batch:
            on_batch(done, total)
    return results
