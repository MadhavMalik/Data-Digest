# Signal Engine — API + built UI in one image.
#
#   docker build -t signal-engine .
#   docker run -p 8000:8000 --env-file .env signal-engine
#
# Two stages so node never ships in the runtime image.

# ---- stage 1: build the React bundle --------------------------------------
FROM node:22-slim AS ui

WORKDIR /build
# Copy manifests first so `npm ci` is cached until dependencies actually change.
COPY web/package.json web/package-lock.json* ./
RUN npm install --silent

COPY web/ ./
# vite.config.ts writes to ../ui/dist, so give it somewhere to land.
RUN mkdir -p /ui && npm run build

# ---- stage 2: runtime ------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SIGNAL_HOST=0.0.0.0 \
    SIGNAL_PORT=8000

WORKDIR /app

# matplotlib needs a writable cache dir when running as a non-root user.
ENV MPLCONFIGDIR=/tmp/mpl

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -q --upgrade pip \
 && pip install -q -e ".[elastic]" \
 && pip install -q python-multipart aiohttp openpyxl pyshp

COPY scripts/ ./scripts/
COPY --from=ui /ui/dist ./ui/dist

# Writable dirs for uploads, generated plots and caches.
RUN mkdir -p data/raw data/metadata data/geo artifacts .cache \
 && useradd -m -u 10001 signal \
 && chown -R signal:signal /app
USER signal

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "signal_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
