# AI Customer Support Agent — container image for the Streamlit app.
# 正式实验可传入带 sha256 digest 的镜像引用，避免基础镜像标签漂移：
# docker build --build-arg PYTHON_IMAGE=python@sha256:<digest> ...
ARG PYTHON_IMAGE=python:3.11.15-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY . .

EXPOSE 8501

# GEMINI_API_KEY (and optionally GEMINI_API_KEYS / GROQ_API_KEY) are provided
# at runtime: docker run -e GEMINI_API_KEY=... -p 8501:8501 support-agent
CMD ["sh", "-c", "python -m src.backend.seed && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
