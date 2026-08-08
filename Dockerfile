# Two stages so the runtime image carries no build toolchain. The AI extras are
# a separate build arg because faster-whisper and sentence-transformers pull in
# CUDA wheels that multiply the image size — a mock-provider demo image has no
# business shipping them.
FROM python:3.12-slim AS build

ARG EXTRAS=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY vaani ./vaani

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".${EXTRAS}"


FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 vaani

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=vaani:vaani vaani ./vaani
COPY --chown=vaani:vaani scripts ./scripts

RUN mkdir -p /app/data /app/models /app/knowledge && chown -R vaani:vaani /app
USER vaani

EXPOSE 8080 9092

# No --log-config: uvicorn rejects an empty file, and vaani.core.logging already
# reconfigures uvicorn's own loggers onto the JSON formatter at startup.
CMD ["uvicorn", "vaani.main:app", "--host", "0.0.0.0", "--port", "8080"]
