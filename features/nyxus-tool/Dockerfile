FROM polusai/bfio:2.5.0

ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"
ENV NUM_WORKERS=4

WORKDIR ${EXEC_DIR}

# Copy the tool
COPY . ${EXEC_DIR}/nyxus-tool

# Upgrade pip/setuptools/wheel first
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install build dependencies (compiler + Python headers)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install your package (Annoy will build successfully)
RUN pip3 install "${EXEC_DIR}/nyxus-tool" --no-cache-dir

# Clean up build tools to reduce image size
RUN apt-get purge -y build-essential python3.11-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Entrypoint
ENTRYPOINT ["python3", "-m", "polus.images.features.nyxus_tool"]
CMD ["--help"]
