FROM --platform=linux/amd64 polusai/bfio:2.5.0

ENV EXEC_DIR="/opt/executables" \
    TOOL_DIR="${EXEC_DIR}/nyxus-tool" \
    POLUS_IMG_EXT=".ome.tif" \
    POLUS_TAB_EXT=".csv" \
    POLUS_LOG="INFO" \
    NUM_WORKERS=4

# Copy source code
COPY . ${TOOL_DIR}
WORKDIR ${TOOL_DIR}

# Upgrade pip tooling
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -e .

# Optional cleanup (safe now)
RUN apt-get purge -y build-essential python3.11-dev || true \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set entrypoint (adjust if your module path differs)
ENTRYPOINT ["python3", "-m", "polus.images.features.nyxus_tool"]
CMD ["--help"]
