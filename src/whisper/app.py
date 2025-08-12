# src/whisper/app.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from faster_whisper import WhisperModel

@dataclass
class ASRConfig:
    model_dir: str | Path
    device: str = "cpu"           # "cpu" | "cuda"
    compute_type: str = "int8"    # cpu: "int8", gpu: "float16"/"int8_float16"
    cpu_threads: int = 8
    language: Optional[str] = "ru"  # None => автоопределение
    beam_size: int = 1
    vad_filter: bool = True
    min_silence_ms: int = 500

class ASR:
    """
    Простой интерфейс:
      asr = ASR(ASRConfig(model_dir="src/whisper/models/faster-whisper-small"))
      out = asr.transcribe("path/to/audio.wav")
    """
    _model: WhisperModel | None = None  # ленивая инициализация

    def __init__(self, model_dir: str | Path, device: str = "cpu",
                 compute_type: str = "int8", cpu_threads: int = 8,
                 language: str | None = "ru", beam_size: int = 1,
                 vad_filter: bool = True, min_silence_ms: int = 500):
        self.cfg = ASRConfig(
            model_dir=model_dir, device=device, compute_type=compute_type,
            cpu_threads=cpu_threads, language=language, beam_size=beam_size,
            vad_filter=vad_filter, min_silence_ms=min_silence_ms
        )

    def _ensure_model(self) -> WhisperModel:
        if ASR._model is None:
            ASR._model = WhisperModel(
                str(self.cfg.model_dir),
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
                cpu_threads=self.cfg.cpu_threads,
            )
        return ASR._model

    def transcribe(self, audio_path: str | Path) -> Dict[str, Any]:
        """
        Возвращает:
          {
            "text": str,
            "segments": [{"start":float,"end":float,"text":str}, ...],
            "lang": str,
            "dur": float,
            "model": {"device":..,"compute_type":..}
          }
        """
        model = self._ensure_model()
        segments, info = model.transcribe(
            str(audio_path),
            language=self.cfg.language,            # None => авто
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
            vad_parameters={"min_silence_duration_ms": self.cfg.min_silence_ms},
        )

        seg_list: List[Dict[str, Any]] = []
        parts: List[str] = []
        for s in segments:
            seg_list.append({"start": s.start, "end": s.end, "text": s.text})
            parts.append(s.text)
        text = " ".join(parts).strip()

        return {
            "text": text,
            "segments": seg_list,
            "lang": info.language,
            "dur": info.duration,
            "model": {
                "device": self.cfg.device,
                "compute_type": self.cfg.compute_type,
                "threads": self.cfg.cpu_threads,
            },
        }

# Утилита для инициализации из config.json (если используешь мой main.py)
def from_config(cfg: dict) -> ASR:
    return ASR(
        model_dir=cfg["whisper_model_dir"],
        device=cfg.get("device", "cpu"),
        compute_type=cfg.get("compute_type", "int8"),
        cpu_threads=int(cfg.get("cpu_threads", 8)),
        language=cfg.get("language", "ru"),
        beam_size=int(cfg.get("beam_size", 1)),
        vad_filter=bool(cfg.get("vad_filter", True)),
        min_silence_ms=int(cfg.get("min_silence_ms", 500)),
    )
