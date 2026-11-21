import os

from dotenv import load_dotenv

load_dotenv()


def _clean(value: str) -> str:
    """Убрать пробелы и кавычки: `docker run --env-file` не снимает кавычки из значений."""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    return v


UNSLOTH_API_BASE = _clean(os.environ.get("UNSLOTH_API_BASE", "")).rstrip("/")
UNSLOTH_API_KEY = _clean(os.environ.get("UNSLOTH_API_KEY", ""))
# LLM для перевода: имя модели из /v1/models инстанса. Без него API берёт модель
# «по умолчанию», а при нескольких загруженных это может оказаться не-LLM (например,
# BS-RoFormer — и тогда chat/completions падает с 400). Пусто = model не передаём.
UNSLOTH_TRANSLATE_MODEL = _clean(os.environ.get("UNSLOTH_TRANSLATE_MODEL", ""))

# Корень приложения (каталог с .env и app/). Относительные пути из .env считаем
# от него, а не от CWD: иначе каталоги «плывут» в зависимости от места запуска uvicorn.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs_dir(value: str, default: str) -> str:
    v = _clean(os.environ.get(value, "")) or default
    return v if os.path.isabs(v) else os.path.normpath(os.path.join(BASE_DIR, v))


# Куда складывать готовые видео — обязательно в .env, дефолта нет: путь зависит от окружения
STORAGE_DIR = _clean(os.environ.get("STORAGE_DIR", ""))
if STORAGE_DIR and not os.path.isabs(STORAGE_DIR):
    STORAGE_DIR = os.path.normpath(os.path.join(BASE_DIR, STORAGE_DIR))
WORK_DIR = _abs_dir("WORK_DIR", os.path.join(BASE_DIR, "work"))


def api_settings_error() -> str | None:
    """Вернуть сообщение об ошибке, если API не настроен в .env."""
    if not UNSLOTH_API_BASE:
        return "UNSLOTH_API_BASE не задан в .env"
    if not UNSLOTH_API_BASE.startswith(("http://", "https://")):
        return f"UNSLOTH_API_BASE должен начинаться с http:// или https:// (сейчас: {UNSLOTH_API_BASE!r})"
    if not UNSLOTH_API_KEY:
        return "UNSLOTH_API_KEY не задан в .env"
    return None
