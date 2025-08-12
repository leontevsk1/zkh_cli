# priorityClassif.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import json
import platform
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from joblib import dump
import sklearn

# === Пути ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = PROJECT_ROOT / "learn/content/importace_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "src/ml/models/urgency_lgbm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === Загрузка данных ===
df = pd.read_csv(DATA_CSV)
X_train, X_test, y_train, y_test = train_test_split(
    df[["Desc", "Group"]],
    df["Priority"],
    test_size=0.2,
    random_state=42
)

# === Преобразования ===
text_transformer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
service_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("Desc", text_transformer, "Desc"),
    ("Group", service_transformer, ["Group"])
])

# === Модель ===
lgb_reg = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", lgb_reg)
])

model.fit(X_train, y_train)

# === Метрики ===
y_pred = np.rint(model.predict(X_test)).clip(1, 4).astype(int)
print("MAE:", mean_absolute_error(y_test, y_pred))

# === Сохранение модели ===
model_path = MODEL_DIR / "urgency_lgbm_pipeline.joblib"
dump(model, model_path)

# === Сохранение метаданных ===
meta = {
    "name": "urgency_lgbm_pipeline",
    "task": "ordinal_regression_1_4",
    "created_at": datetime.utcnow().isoformat() + "Z",
    "versions": {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "lightgbm": lgb.__version__
    },
    "postprocess": {"clip": [1, 4], "round": True},
    "inputs": {"text": "str", "service": "str"}
}
meta_path = MODEL_DIR / "urgency_meta.json"
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"✅ Saved: {model_path.name}, {meta_path.name}")
