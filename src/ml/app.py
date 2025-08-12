from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, time, json
import numpy as np
import torch
import joblib

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel
from huggingface_hub import snapshot_download

# ===== Контракты =====
@dataclass
class InferenceOut:
    service: str
    service_prob: float
    urgency: str | int
    urgency_score: float
    summary: str | None
    meta: dict

# ===== Пути к моделям =====
ML_MODELS_ROOT = Path(__file__).resolve().parent / "models"
ALT_MODULES_ROOT = Path(__file__).resolve().parents[2] / "modules"
MODELS_DIR = ML_MODELS_ROOT if ML_MODELS_ROOT.exists() else ALT_MODULES_ROOT

SVC_DIR = MODELS_DIR / "service_clf"
URG_DIR = MODELS_DIR / "urgency_lgbm"

SERVICE_CLF_PATH  = SVC_DIR / "service_clf_pipeline.joblib"   # теперь тут весь пайплайн
SERVICE_META_PATH = SVC_DIR / "service_meta.json"

URGENCY_PIPE_PATH = URG_DIR / "urgency_lgbm_pipeline.joblib"
URGENCY_META_PATH = URG_DIR / "urgency_meta.json"

ADAPTER_DIR = MODELS_DIR / "zkh_problem_lora"
USE_LORA = ADAPTER_DIR.exists()

# ===== Базовая модель для суммаризации =====
base_model_name = os.getenv("PROBLEM_BASE_MODEL", None)
if USE_LORA:
    peft_cfg = PeftConfig.from_pretrained(ADAPTER_DIR)
    base_model_name = base_model_name or peft_cfg.base_model_name_or_path
if not base_model_name:
    base_model_name = "cointegrated/rut5-small"

BASE_CACHE_DIR = MODELS_DIR / "base_models" / base_model_name.replace("/", "__")

os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

def ensure_base_model_local(repo_id: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    if (local_dir / "config.json").exists():
        return local_dir
    last_err = None
    for attempt in range(3):
        try:
            p = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=[
                    "*.json", "*.safetensors", "*.bin",
                    "tokenizer.*", "spiece.model", "vocab.*", "merges.txt"
                ],
            )
            return Path(p)
        except Exception as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Base model {repo_id} not found offline in {local_dir}. Last error: {last_err}")

def _dummy_clf(class_labels):
    class _DummyClf:
        def __init__(self, labels):
            self.classes_ = np.array(labels)
        def predict_proba(self, X):
            n = len(X)
            probs = np.zeros((n, len(self.classes_)), dtype=float)
            probs[:, 0] = 1.0
            return probs
        def predict(self, X):
            return np.array([self.classes_[0]] * len(X))
    return _DummyClf(class_labels)


def _safe_joblib_load(p: Path):
    try:
        return joblib.load(p)
    except Exception:
        return None

def _load_json_or_default(path_json: Path, default: dict):
    try:
        if path_json.exists():
            return json.loads(path_json.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def _load_service_pipeline() -> object:
    obj = _safe_joblib_load(SERVICE_CLF_PATH)
    if obj is not None and hasattr(obj, "predict_proba"):
        return obj
    return _dummy_clf(["other"])

def _load_urgency_pipeline() -> object:
    obj = _safe_joblib_load(URGENCY_PIPE_PATH)
    if obj is not None and hasattr(obj, "predict"):
        return obj
    return _dummy_clf([1, 2, 3, 4])


# ===== Основной класс =====
class Models:
    def __init__(self, model_root: str | Path | None = None, device: str = "cpu"):
        self.service_clf = _load_service_pipeline()
        self.svc_meta    = _load_json_or_default(SERVICE_META_PATH, {"version": "unknown", "source": "unknown"})

        self.urgency_clf = _load_urgency_pipeline()
        self.urg_meta    = _load_json_or_default(URGENCY_META_PATH, {"version": "unknown", "source": "unknown"})

        self.local_base = ensure_base_model_local(base_model_name, BASE_CACHE_DIR)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_base, use_fast=True, legacy=False)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_base, use_fast=False)

        if ADAPTER_DIR.exists():
            base = AutoModelForSeq2SeqLM.from_pretrained(self.local_base, torch_dtype=torch.float32)
            self.summarizer = PeftModel.from_pretrained(base, ADAPTER_DIR)
            self.sum_meta = {"type": "lora", "base_model": str(self.local_base), "adapter_dir": str(ADAPTER_DIR)}
        else:
            self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(self.local_base, torch_dtype=torch.float32)
            self.sum_meta = {"type": "base", "base_model": str(self.local_base), "adapter_dir": None}
        self.summarizer.eval()
        self.device = device
        self.summarizer.to("cpu")

    @torch.inference_mode()
    def _summarize(self, text: str, max_new_tokens: int = 50) -> str:
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
        out = self.summarizer.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _preprocess(self, text: str) -> str:
        return text.strip()

    def infer(self, text: str) -> InferenceOut:
        x = [text]

        svc_proba = self.service_clf.predict_proba(x)[0]
        svc_idx = int(np.argmax(svc_proba))
        svc_label = str(self.service_clf.classes_[svc_idx])

        # 💡 Новый блок для регрессора
        urg_pred = self.urgency_clf.predict(x)[0]
        urg_label = int(np.clip(np.rint(urg_pred), 1, 4))

        summary = self._summarize(text, max_new_tokens=50)

        return InferenceOut(
            service=svc_label,
            service_prob=float(svc_proba[svc_idx]),
            urgency=urg_label,
            urgency_score=float(urg_pred),
            summary=summary,
            meta={
                "svc_meta": self.svc_meta,
                "urgency_meta": self.urg_meta,
                "summarizer": self.sum_meta
            }
        )

