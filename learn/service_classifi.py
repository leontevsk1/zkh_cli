# servclassification.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import re, json
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline, FeatureUnion
import joblib

from src.ml.utils.rule_features import RuleFeatures  # <-- 💡 отдельно, уже есть

# === Пути ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = PROJECT_ROOT / "learn/content/L_corrected.csv"
OUT_DIR = PROJECT_ROOT / "src/ml/models/service_clf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "Desc"
LABEL_COL = "Group"
MAJOR_CLASS = "Управление домом"

# === Лексиконы ===
LEX: Dict[str, List[str]] = {
    "Канализация": [r"\bканализац", r"\bзасор", r"\bпрочистк", r"\bколодец", r"\bстояк", r"\bсифон", r"\bунитаз", r"\bраковин", r"\bслив", r"\bфанова", r"\bзапах(?!.*газа)", r"\bжироулав", r"\bкрышка колодца"],
    "Водоотведение": [r"\bводоотвед", r"\bкнс\b", r"\bливнев", r"\bдождепри", r"\bстоки", r"\bфекаль", r"\bзатоп(ил[аи]|ило)\b", r"\bподвале\b.*\bвода\b", r"\bперелив"],
    "Электроэнергия": [r"\bнет\s+света", r"\bвыбил[аи].*\bпробк", r"\bпробк[аи]\b", r"\bавтомат(ы)?\b", r"\bщиток", r"\bзамыкан", r"\bкоротит", r"\bискрит", r"\bпровод", r"\bрозетк", r"\bосвещен", r"\bламп(а|ы|очек|очку)"],
    "ГВС": [r"\bгвс\b", r"\bгоряч(ая|ей)\s+вод", r"\bнет\s+горяч", r"\bтемператур", r"\bсчетчик", r"\bопломб", r"\bпломб", r"\bподмес", r"\bциркуляц"],
    "Лифты": [r"\bлифт", r"\bзастрял", r"\bзастряли", r"\bне\s+работает\s+лифт", r"\bдвер(ь|и)\s+(не\s+)?(закрыва|открыва)", r"\bкнопк[аи]\s+вызова", r"\bкабин[ае]", r"\bскрежет", r"\bреверс"],
    "Домофоны": [r"\bдомофон", r"\bтрубк", r"\bпанел", r"\bвызов", r"\bне\s+работает\s+домофон", r"\bключ(и|ей)\b", r"\bмагнит", r"\bдвер(ь|и)\b.*\bне\s+открыва", r"\bкод(ы)?", r"\bзвонок"],
    "Управление домом": [r"\bподъезд", r"\bдвор", r"\bуборк[аи]", r"\bмусор", r"\bснег", r"\bналедь|\bгололед", r"\bкровл|крыша", r"\bдвер(ь|и)\s+подъезда", r"\bперила", r"\bпочистить"],
    "Отопление": [r"\bнет\s+отоплен", r"\bбатаре(я|и)\s+(холод|лед)", r"\bрадиатор", r"\bстояк\s+отоплен", r"\bкотел"],
    "Газоснабжение": [r"\bгаз\b", r"\bзапах\s+газа", r"\bутечк[аи]\s+газа", r"\bгазов", r"\bпломб.*газ"]
}

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

# === Данные ===
df = pd.read_csv(DATA_CSV)
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str).map(normalize_text)

X_train, X_val, y_train, y_val = train_test_split(
    df[TEXT_COL].values, df[LABEL_COL].values, test_size=0.2, stratify=df[LABEL_COL], random_state=42
)

# === Фичи ===
classes_sorted = sorted(pd.unique(df[LABEL_COL]))
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)

# === Объединяем признаки ===
features = FeatureUnion([
    ("tfidf", tfidf),
    ("rules", RuleFeatures(class_labels=classes_sorted))  # <-- rule-based признаки
])

# === Классификатор ===
clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)

# === Pipeline ===
pipeline = Pipeline([
    ("features", features),
    ("clf", clf)
])

# === Обучение ===
sample_weight = np.ones(len(y_train), dtype=np.float32)
sample_weight[np.array(y_train) == MAJOR_CLASS] = 0.6
pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight)

# === Предсказания и отчёт ===
y_pred = pipeline.predict(X_val)
print("\n=== Без порога ===")
print(classification_report(y_val, y_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred, labels=clf.classes_))

# === Экспорт модели ===
joblib.dump(pipeline, OUT_DIR / "service_clf.joblib")

# === Экспорт метаданных ===
meta = {
    "version": "1.1",
    "source": "servclassification.py",
    "feature_union": ["TfidfVectorizer", "RuleFeatures"],
    "num_classes": len(classes_sorted)
}
(OUT_DIR / "service_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"✅ Экспорт завершён: {OUT_DIR}")