FROM polusai/bfio:2.5.0

# Environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"
ENV NUM_THREADS=8

WORKDIR ${EXEC_DIR}

ENV TOOL_DIR="transforms/images/ftl-label-tool"

# Install system dependencies:
# - python3.11-dev: Python headers required to compile C/C++ extensions (Python.h)
# - build-essential: gcc/g++ compiler
# - curl: needed to download rustup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11-dev \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the repository
RUN mkdir -p image-tools
COPY . ${EXEC_DIR}/image-tools

# Install build dependencies
RUN pip3 install \
        "setuptools>=68.0" \
        "setuptools-rust>=1.9.0" \
        "cython>=3.0.11" \
        "numpy>2.0.0" \
        wheel \
        --no-cache-dir

# Install the tool (compiles Cython + Rust extensions)
RUN pip3 install "${EXEC_DIR}/image-tools/${TOOL_DIR}" \
        --no-build-isolation \
        --no-cache-dir

ENTRYPOINT ["python3", "-m", "polus.images.transforms.images.ftl_label"]
CMD ["--help"]
