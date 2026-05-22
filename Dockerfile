FROM python:3.12-slim-bookworm

# Системные зависимости для LightGBM и numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости устанавливаем раньше копирования кода — кеш слоёв
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pred_service/ ./pred_service/

# Непривилегированный пользователь (совпадает с run_as_user=1000 в DAG)
RUN useradd -u 1000 -m appuser
USER 1000

CMD ["python", "-m", "pred_service"]
