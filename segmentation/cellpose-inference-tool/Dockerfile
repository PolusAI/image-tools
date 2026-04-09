FROM polusai/bfio:2.5.0

# Environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"
ENV NUM_WORKERS=1

# GPU device selection — override at runtime with -e CUDA_VISIBLE_DEVICES=0
ENV CUDA_VISIBLE_DEVICES=0

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

ENV TOOL_DIR="segmentation/cellpose-inference-tool"

# Install PyTorch:
#   - x86_64 → CUDA 12.4 wheel (requires NVIDIA driver >= 525.60.13 at runtime
#               via NVIDIA Container Toolkit; run with `docker run --gpus all`)
#   - ARM64  → standard wheel (CPU only inside Docker; MPS is not accessible
#               from a Linux VM on macOS — run the tool natively for M-series GPU)
RUN arch="$(uname -m)" && \
    if [ "$arch" = "x86_64" ]; then \
        pip3 install torch --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir; \
    else \
        pip3 install torch --no-cache-dir; \
    fi

# Copy the repository into the container
RUN mkdir image-tools
COPY . ${EXEC_DIR}/image-tools

# Install the tool (pulls in cellpose, bfio, filepattern, typer, tqdm …)
RUN pip3 install "${EXEC_DIR}/image-tools/${TOOL_DIR}" --no-cache-dir

# Set the entrypoint
ENTRYPOINT ["python3", "-m", "polus.images.segmentation.cellpose_inference"]
CMD ["--help"]
