import re
import numpy as np
from typing import List, Dict

# Словарь ключевых паттернов для классов
LEX: Dict[str, List[str]] = {
    "Канализация": [
        r"\bканализац", r"\bзасор", r"\bпрочистк", r"\bколодец", r"\bстояк", r"\bсифон",
        r"\bунитаз", r"\bраковин", r"\bслив", r"\bфанова", r"\bзапах(?!.*газа)", r"\bжироулав", r"\bкрышка колодца"
    ],
    "Водоотведение": [
        r"\bводоотвед", r"\bкнс\b", r"\bливнев", r"\bдождепри", r"\bстоки", r"\bфекаль",
        r"\bзатоп(ил[аи]|ило)\b", r"\bподвале\b.*\bвода\b", r"\bперелив"
    ],
    "Электроэнергия": [
        r"\bнет\s+света", r"\bвыбил[аи].*\bпробк", r"\bпробк[аи]\b", r"\bавтомат(ы)?\b", r"\bщиток",
        r"\bзамыкан", r"\bкоротит", r"\bискрит", r"\bпровод", r"\bрозетк", r"\bосвещен", r"\bламп(а|ы|очек|очку)"
    ],
    "ГВС": [
        r"\bгвс\b", r"\bгоряч(ая|ей)\s+вод", r"\bнет\s+горяч", r"\bтемператур", r"\bсчетчик", r"\bопломб",
        r"\bпломб", r"\bподмес", r"\bциркуляц"
    ],
    "Лифты": [
        r"\bлифт", r"\bзастрял", r"\bзастряли", r"\bне\s+работает\s+лифт", r"\bдвер(ь|и)\s+(не\s+)?(закрыва|открыва)",
        r"\bкнопк[аи]\s+вызова", r"\bкабин[ае]", r"\bскрежет", r"\bреверс"
    ],
    "Домофоны": [
        r"\bдомофон", r"\bтрубк", r"\bпанел", r"\bвызов", r"\bне\s+работает\s+домофон", r"\bключ(и|ей)\b",
        r"\bмагнит", r"\bдвер(ь|и)\b.*\bне\s+открыва", r"\bкод(ы)?", r"\bзвонок"
    ],
    "Управление домом": [
        r"\bподъезд", r"\bдвор", r"\bуборк[аи]", r"\bмусор", r"\bснег", r"\bналедь|\bгололед", r"\bкровл|крыша",
        r"\bдвер(ь|и)\s+подъезда", r"\bперила", r"\bпочистить"
    ],
    "Отопление": [
        r"\bнет\s+отоплен", r"\bбатаре(я|и)\s+(холод|лед)", r"\bрадиатор", r"\bстояк\s+отоплен", r"\bкотел"
    ],
    "Газоснабжение": [
        r"\bгаз\b", r"\bзапах\s+газа", r"\bутечк[аи]\s+газа", r"\bгазов", r"\bпломб.*газ"
    ],
}

def build_rule_feats(texts: List[str], classes: List[str]) -> np.ndarray:
    comp = {c: [re.compile(p, re.I) for p in LEX.get(c, [])] for c in classes}
    M = np.zeros((len(texts), len(classes)), dtype=np.float32)
    for i, t in enumerate(texts):
        for j, c in enumerate(classes):
            if comp[c]:
                M[i, j] = sum(1 for pat in comp[c] if pat.search(t))
    return M

class RuleFeatures:
    def __init__(self, class_labels: List[str]):
        self.class_labels = class_labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return build_rule_feats(X, self.class_labels)
