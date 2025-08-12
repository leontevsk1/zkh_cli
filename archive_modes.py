import tarfile
from pathlib import Path

OUTPUT = "models.tar.gz"
INCLUDE = [
    "src/ml/models/service_clf",
    "src/ml/models/urgency_lgbm",
    "src/ml/models/base_models",
    "src/ml/models/zkh_problem_lora",
    "src/whisper/models/faster-whisper-small"
]

def archive():
    with tarfile.open(OUTPUT, "w:gz") as tar:
        for path in INCLUDE:
            path = Path(path)
            if not path.exists():
                print(f"❌ Пропущено (не найдено): {path}")
                continue
            print(f"✅ Добавление: {path}")
            tar.add(path, arcname=path.relative_to("src"))

    print(f"🎉 Архив готов: {OUTPUT}")

if __name__ == "__main__":
    archive()
