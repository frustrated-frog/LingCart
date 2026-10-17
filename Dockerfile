# LingCart — build the React console, then run one FastAPI process.
# 正式实验可传入带 sha256 digest 的镜像引用，避免基础镜像标签漂移：
# docker build --build-arg PYTHON_IMAGE=python@sha256:<digest> ...
ARG PYTHON_IMAGE=python:3.11.15-slim

FROM node:22-alpine AS web-builder

WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY . .
COPY --from=web-builder /web/dist /app/frontend/dist

EXPOSE 8000

# GEMINI_API_KEY (and optionally GEMINI_API_KEYS / GROQ_API_KEY) are provided
# at runtime: docker run -e GEMINI_API_KEY=... -p 8000:8000 lingcart
CMD ["sh", "-c", "python -m src.backend.seed && uvicorn api:app --host=0.0.0.0 --port=8000"]
