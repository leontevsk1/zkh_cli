import subprocess
import sys
import os
from pathlib import Path
import tarfile

REPO_URL = "https://github.com/leontevsk1/zkh_cli.git"
REPO_NAME = "zkh_cli"
MODELS_URL = "https://github.com/leontevsk1/zkh_cli/releases/download/%23models/models.tar.gz"
ARCHIVE_NAME = "models.tar.gz"
INSTALL_DIR = Path.cwd() / REPO_NAME
MODELS_DIR = INSTALL_DIR / "src" / "ml" / "models"

def run(cmd: str):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"❌ Команда завершилась с ошибкой: {cmd}")

def clone_repo():
    if INSTALL_DIR.exists():
        print(f"✅ Репозиторий уже существует: {INSTALL_DIR}")
    else:
        run(f"git clone {REPO_URL}")

def create_venv():
    run(f"python -m venv {INSTALL_DIR / '.venv'}")
    print("✅ Виртуальное окружение создано")

def install_dependencies():
    pip = INSTALL_DIR / ".venv" / "bin" / "pip"
    run(f"{pip} install --upgrade pip")
    run(f"{pip} install -r {INSTALL_DIR / 'req.text'}")

def download_models():
    print(f"📥 Скачиваем модели из: {MODELS_URL}")
    run(f"curl -L -o {ARCHIVE_NAME} {MODELS_URL}")

def extract_models():
    print("📦 Распаковываем архив моделей...")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall(path=MODELS_DIR.parent)  # ← Архив должен содержать `models/` внутри
    print(f"✅ Модели извлечены в: {MODELS_DIR}")

def main():
    clone_repo()
    create_venv()
    install_dependencies()
    download_models()
    extract_models()
    print("🎉 Установка завершена. Для запуска активируйте виртуальное окружение:")
    print(f"source {INSTALL_DIR / '.venv' / 'bin' / 'activate'}")

if __name__ == "__main__":
    main()
