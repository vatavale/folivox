# Folivox studio in a container. One Dockerfile, three builds (see docs/docker.md):
#   cpu       docker build -f docker/Dockerfile --build-arg VARIANT=cpu -t folivox:cpu .
#   cuda      docker build -f docker/Dockerfile --build-arg VARIANT=cuda -t folivox:cuda .       (torch 2.5, CUDA 12.4)
#   cuda128   docker build -f docker/Dockerfile --build-arg VARIANT=cuda128 -t folivox:cuda128 . (RTX 50: torch 2.7.1, CUDA 12.8)
# Models (about 7.6 GB) are NOT in the image: they are downloaded into the /data volume on first use.
ARG PYTHON_VERSION=3.11

# trixie, not bookworm: the native audio.cpp build needs glibc 2.38+ (bookworm has 2.36)
FROM python:${PYTHON_VERSION}-slim-trixie AS build
ARG VARIANT=cpu
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /src
SHELL ["/bin/bash", "-c"]
COPY pyproject.toml README.md LICENSE NOTICE ./
# The same torch pins as scripts/install.sh. cuda: the default PyPI wheels already carry their CUDA libraries.
RUN set -eu; \
    case "$VARIANT" in \
      cpu)     TORCH="torch<2.6";     EXTRA="--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu" ;; \
      cuda)    TORCH="torch<2.6";     EXTRA="" ;; \
      cuda128) TORCH="torch==2.7.1";  EXTRA="--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128" ;; \
      *) echo "unknown VARIANT $VARIANT" >&2; exit 1 ;; \
    esac; \
    printf '%s\n%s\n' "$TORCH" "${TORCH/torch/torchaudio}" > /tmp/constraints.txt; \
    uv venv /opt/folivox; \
    uv pip install --python /opt/folivox/bin/python $EXTRA --constraints /tmp/constraints.txt \
      -r pyproject.toml --extra tts --extra qa --extra ui --extra denoise --extra anthropic
# The heavy layer above depends on pyproject.toml only; a source change reinstalls just the package below.
COPY src ./src
RUN uv pip install --python /opt/folivox/bin/python --no-deps .

FROM python:${PYTHON_VERSION}-slim-trixie
ARG VARIANT=cpu
LABEL org.opencontainers.image.source="https://github.com/vatavale/folivox" \
      org.opencontainers.image.description="Folivox studio: e-books to chaptered audiobooks, locally (${VARIANT})" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini libsndfile1 libvulkan1 util-linux ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd -g 1000 folivox && useradd -u 1000 -g 1000 -m -d /home/folivox -s /usr/sbin/nologin folivox \
 && mkdir -p /data /audiobooks /work && chown folivox:folivox /data /audiobooks /work
COPY --from=build /opt/folivox /opt/folivox
COPY docker/entrypoint.sh /usr/local/bin/folivox-entrypoint
RUN chmod 755 /usr/local/bin/folivox-entrypoint
# Everything Folivox writes lives under /data (models, voices, settings, studio state, token file, Hugging Face cache).
ENV PATH=/opt/folivox/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    XDG_DATA_HOME=/data/share \
    XDG_CONFIG_HOME=/data/config \
    XDG_CACHE_HOME=/data/cache \
    HF_HOME=/data/cache/huggingface \
    FOLIVOX_OUTPUT_DIR=/audiobooks \
    FOLIVOX_WORK_DIR=/work \
    FOLIVOX_UI_HOST=0.0.0.0 \
    FOLIVOX_UI_PORT=8765 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
VOLUME ["/data", "/audiobooks", "/work"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request as u;u.urlopen('http://127.0.0.1:%s/healthz'%os.environ.get('FOLIVOX_UI_PORT','8765'),timeout=4)" || exit 1
ENTRYPOINT ["tini", "-g", "--", "folivox-entrypoint"]
CMD ["folivox", "ui"]
