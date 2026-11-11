# Build from router/:  docker build -f anthropic_masker/Dockerfile -t pii-masker .
FROM python:3.12-slim
COPY anthropic_masker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && useradd -u 10002 -M masker
WORKDIR /app
# Only the masking modules: the router's LiteLLM hook and admin are not needed here.
COPY pii_router/__init__.py pii_router/detectors.py pii_router/mask.py ./pii_router/
COPY inference_router/__init__.py inference_router/config.py inference_router/detection.py ./inference_router/
COPY anthropic_masker/__init__.py anthropic_masker/app.py ./anthropic_masker/
USER 10002
EXPOSE 8080
CMD ["uvicorn", "--factory", "anthropic_masker.app:app_factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
