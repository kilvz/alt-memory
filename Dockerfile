# alt-memory — production Docker image
# Uses python:3.12-slim for minimal footprint.
# Optional: --build-arg EXTRAS=onnx,chroma,all  (default: no extras)
#
# Build:
#   docker build -t alt-memory .
#   docker build --build-arg EXTRAS=all -t alt-memory:latest .
#
# Run:
#   docker run -v ~/.alt-memory:/root/.alt-memory alt-memory --help
#   docker run -v ~/.alt-memory:/root/.alt-memory alt-memory-mcp

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ARG EXTRAS=""
RUN pip install --no-cache-dir .${EXTRAS:+"[$EXTRAS]"}

# ── Runtime stage ──────────────────────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/alt-memory /usr/local/bin/alt-memory
COPY --from=builder /usr/local/bin/alt-memory-mcp /usr/local/bin/alt-memory-mcp

VOLUME ["/root/.alt-memory"]
EXPOSE 8100

ENTRYPOINT ["alt-memory"]
CMD ["--help"]
