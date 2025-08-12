#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

# Импорты из модулей пайплайна
from whisper.app import ASR
from ml.app import Models
from users_messages.app import review_and_confirm

def parse_args():
    p = argparse.ArgumentParser("ЖКХ: обработка голосовых обращений")
    p.add_argument(
        "--audio",
        help="Путь к аудиофайлу (wav/mp3/m4a/ogg и т.п.)",
        required=True
    )
    p.add_argument(
        "--config",
        default="src/config.json",
        help="Путь к JSON-файлу с настройками пайплайна"
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Автоподтверждение (не спрашивать y/N)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"❌ Config not found: {cfg_path}")
        sys.exit(1)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # === Whisper ===
    asr = ASR(
        model_dir=cfg["whisper_model_dir"],
        device=cfg.get("device", "cpu"),
        compute_type=cfg.get("compute_type", "int8"),
        cpu_threads=int(cfg.get("cpu_threads", 8)),
        language=cfg.get("language", "ru"),
        beam_size=int(cfg.get("beam_size", 1)),
        vad_filter=bool(cfg.get("vad_filter", True)),
        min_silence_ms=int(cfg.get("min_silence_ms", 500)),
    )

    print("🎙 Распознавание речи...")
    asr_out = asr.transcribe(args.audio)
    if not asr_out["text"]:
        print("❌ Текст не распознан. Завершение.")
        sys.exit(2)

    # === ML ===
    nlp = Models(cfg["ml_models_root"])
    print("🤖 Классификация и суммаризация...")
    ml_out = nlp.infer(asr_out["text"])

    # === Подтверждение ===
    saved = review_and_confirm(ml_out, auto_confirm=args.yes)
    if saved:
        print("✅ Заявка сохранена.")
    else:
        print("🚫 Заявка не сохранена.")

if __name__ == "__main__":
    main()
