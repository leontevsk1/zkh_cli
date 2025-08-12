# src/users_messages/app.py
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from typing import Optional
import json
import time
import shutil
import sys

DEFAULT_STORE = Path("zkh-pipeline/messages.jsonl")

def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default

def _hr(ch: str = "─") -> str:
    return ch * max(20, min(_term_width(), 120))

def _fmt_prob(p: float) -> str:
    return f"{p*100:.1f}%" if p is not None else "—"

def _truncate(s: str, n: int = 220) -> str:
    s = s.strip()
    return s if len(s) <= n else (s[:n] + "…")

def build_readable(ml_out) -> str:
    """
    Формирует текстовое представление результата ML.
    ml_out — InferenceOut или dict с ключами:
      service, service_prob, urgency, urgency_score, summary, meta
    """
    mld = asdict(ml_out) if hasattr(ml_out, "__dataclass_fields__") else dict(ml_out)

    service        = mld.get("service")
    service_prob   = mld.get("service_prob")
    urgency        = mld.get("urgency")
    urgency_score  = mld.get("urgency_score")
    summary        = mld.get("summary")
    meta           = mld.get("meta", {})

    svc_meta = meta.get("svc_meta", {})
    urg_meta = meta.get("urgency_meta", {})
    sum_meta = meta.get("summarizer", {})

    parts = []
    parts.append(_hr())
    parts.append("Результаты классификации ЖКХ:")
    parts.append(_hr())
    parts.append(f"• Служба: {service}  (доверие { _fmt_prob(service_prob) })")
    parts.append(f"• Срочность: {urgency}  (оценка {urgency_score:.2f})")
    if summary:
        parts.append("• Краткое описание:")
        parts.append(_truncate(summary, 600))
    parts.append(_hr())
    parts.append("Модели:")
    parts.append(f"• service_clf: {svc_meta.get('version','?')} ({svc_meta.get('source','?')})")
    parts.append(f"• urgency_clf: {urg_meta.get('version','?')} ({urg_meta.get('source','?')})")
    parts.append(f"• summarizer: {sum_meta.get('type','?')} / {Path(str(sum_meta.get('base_model',''))).name or '?'}")
    parts.append(_hr())
    parts.append("Отправить пользователю и сохранить в хранилище? [y/N]: ")
    return "\n".join(parts)

def _input_yes_no(prompt_line: str = "") -> bool:
    if prompt_line:
        sys.stdout.write(prompt_line)
        sys.stdout.flush()
    ans = input().strip().lower()
    return ans in ("y", "yes", "д", "да")

def _ensure_store(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()

def save_record(
    store_path: Path,
    ml_out,
    extra: Optional[dict] = None
) -> Path:
    """
    Пишет одну запись JSONL. Возвращает путь к файлу.
    """
    rec = {
        "ts": int(time.time()),
        "ml": asdict(ml_out) if hasattr(ml_out, "__dataclass_fields__") else dict(ml_out),
    }
    if extra:
        rec["extra"] = extra

    _ensure_store(store_path)
    with store_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return store_path

def review_and_confirm(
    ml_out,
    store_path: Path = DEFAULT_STORE,
    auto_confirm: bool = False
) -> bool:
    """
    1) Формирует сводку.
    2) Спрашивает подтверждение y/N (или подтверждает автоматически).
    3) При 'y' — сохраняет запись в JSONL.
    Возвращает True, если запись сохранена.
    """
    view = build_readable(ml_out)
    print(view, end="")  # последний перевод строки уже есть
    ok = True if auto_confirm else _input_yes_no()
    if not ok:
        print("Отменено пользователем.")
        return False

    save_record(store_path, ml_out)
    print(f"Сохранено: {store_path}")
    return True
