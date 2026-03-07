# Этап 1: Сборка зависимостей
FROM python:3.11-slim as builder

WORKDIR /app

# Установка системных зависимостей для сборки (если потребуются для pip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем только файл зависимостей для кеширования слоев
COPY req.text .
RUN pip install --no-cache-dir --prefix=/install -r req.text

# Этап 2: Финальный образ
FROM python:3.11-slim

WORKDIR /app

# Установка ffmpeg (критично для Whisper/ASR) и curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем установленные библиотеки из билдера
COPY --from=builder /install /usr/local

# Копируем исходный код проекта
COPY src/ ./src/

# Создаем директории для моделей и логов
RUN mkdir -p src/ml/models src/whisper/models zkh-pipeline

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# По умолчанию запускаем main.py, но оставляем возможность передать путь к аудио
ENTRYPOINT ["python", "src/main.py"]
CMD ["--help"]
