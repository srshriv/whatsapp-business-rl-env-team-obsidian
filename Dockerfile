FROM python:3.11-slim

LABEL maintainer="Dev A"
LABEL description="WhatsApp Sales RL – OpenEnv server"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        nginx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

RUN chmod +x start.sh

CMD ["bash", "start.sh"]