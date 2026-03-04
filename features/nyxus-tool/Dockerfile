FROM polusai/bfio:2.4.5

ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

WORKDIR ${EXEC_DIR}

# Copy the tool
COPY . ${EXEC_DIR}/nyxus-tool

# Install build dependencies — compiler + Python headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    python3-dev \              
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel (good practice)
RUN python3 -m pip install --upgrade pip setuptools wheel

# Now install your package — annoy should build successfully
RUN pip3 install "${EXEC_DIR}/nyxus-tool" --no-cache-dir

# Clean up build tools to reduce image size
RUN apt-get purge -y --auto-remove \
    g++ \
    gcc \
    python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["python3", "-m", "polus.images.features.nyxus_tool"]
CMD ["--help"]