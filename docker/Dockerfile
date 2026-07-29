# =============================================================================
# Build-Optimization Studio (Tauri) — Linux bundle build container
# =============================================================================
# Builds the Studio desktop application (app/) into distributable Linux
# bundles (.deb + .AppImage). Extract the artifacts with:
#
#   docker build -f docker/Dockerfile --output type=local,dest=dist-studio .
#
# The final stage contains only the built bundles.

FROM rust:1-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Tauri 2.0 Linux build dependencies (WebKitGTK 4.1) + Node.js 22
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libwebkit2gtk-4.1-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libssl-dev \
    file \
    wget \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install frontend dependencies first (better layer caching)
COPY app/package.json app/package-lock.json* ./app/
RUN cd app && npm install

# Copy the Studio sources and build the bundles
COPY app ./app
RUN cd app && npm run tauri:build:linux

# -----------------------------------------------------------------------------
# Final stage: bundles only
# -----------------------------------------------------------------------------
FROM scratch AS bundles

COPY --from=builder /build/app/src-tauri/target/release/bundle/deb/*.deb /
COPY --from=builder /build/app/src-tauri/target/release/bundle/appimage/*.AppImage /
