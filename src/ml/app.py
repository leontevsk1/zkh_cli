from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, time, json
import numpy as np
import torch
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftConfig, PeftModel
from huggingface_hub import snapshot_download
from ml.utils.rule_features import RuleFeatures


@dataclass
class InferenceOut:
    service: str
    service_prob: float
    urgency: str | int
    urgency_score: float
    summary: str | None
    meta: dict


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


class Models:
    def __init__(self, model_root: str | Path | None = None, device: str = "cpu"):
        self.device = device
        self.model_root = Path(model_root).resolve() if model_root else Path(__file__).resolve().parent / "models"

        svc_dir = self.model_root / "service_clf"
        urg_dir = self.model_root / "urgency_lgbm"
        adapter_dir = self.model_root / "zkh_problem_lora"
        base_cache_dir = self.model_root / "base_models"
        
        svc_path = svc_dir / "service_clf.joblib"
        loaded_svc = _safe_joblib_load(svc_path)
        if loaded_svc is not None and hasattr(loaded_svc, "predict_proba"):
            self.service_clf = loaded_svc
        else:
            print(f"⚠️  Failed to load valid service_clf from {svc_path}, using dummy.")
            self.service_clf = _dummy_clf(["other"])


        self.svc_meta = _load_json_or_default(svc_dir / "service_meta.json", {"version": "unknown", "source": "unknown"})

        self.urgency_clf = _safe_joblib_load(urg_dir / "urgency_lgbm_pipeline.joblib") or _dummy_clf([1, 2, 3, 4])
        self.urg_meta = _load_json_or_default(urg_dir / "urgency_meta.json", {"version": "unknown", "source": "unknown"})

        base_model_name = os.getenv("PROBLEM_BASE_MODEL", None)
        if adapter_dir.exists():
            peft_cfg = PeftConfig.from_pretrained(adapter_dir)
            base_model_name = base_model_name or peft_cfg.base_model_name_or_path
        if not base_model_name:
            base_model_name = "cointegrated/rut5-small"

        local_base_dir = base_cache_dir / base_model_name.replace("/", "__")
        self.local_base = ensure_base_model_local(base_model_name, local_base_dir)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_base, use_fast=True, legacy=False)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_base, use_fast=False)

        if adapter_dir.exists():
            base = AutoModelForSeq2SeqLM.from_pretrained(self.local_base, torch_dtype=torch.float32)
            self.summarizer = PeftModel.from_pretrained(base, adapter_dir)
            self.sum_meta = {"type": "lora", "base_model": str(self.local_base), "adapter_dir": str(adapter_dir)}
        else:
            self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(self.local_base, torch_dtype=torch.float32)
            self.sum_meta = {"type": "base", "base_model": str(self.local_base), "adapter_dir": None}

        self.summarizer.eval()
        self.summarizer.to(self.device)
        print(f"\n✅ service_clf loaded: {type(self.service_clf)}")
        print(f"  Классы: {getattr(self.service_clf, 'classes_', '?')}")
        print(f"  Метаданные: {self.svc_meta}")

    @torch.inference_mode()
    def _summarize(self, text: str, max_new_tokens: int = 50) -> str:
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
        out = self.summarizer.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def infer(self, text: str) -> InferenceOut:
        x_text = text.strip().lower()
        x_svc = [x_text]
        svc_proba = self.service_clf.predict_proba(x_svc)[0]
        svc_idx = int(np.argmax(svc_proba))
        svc_label = str(self.service_clf.classes_[svc_idx])

        x_urg = pd.DataFrame([{
            "Desc": x_text,
            "Group": svc_label
        }])

        urg_pred = self.urgency_clf.predict(x_urg)[0]
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