FROM polusai/bfio:2.3.3

ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

WORKDIR ${EXEC_DIR}

# Build context = repo root (PolusAI-image-tools). Path is transforms/images/apply-flatfield-tool
COPY transforms/images/apply-flatfield-tool /opt/executables/app

RUN pip3 install --no-cache-dir /opt/executables/app

ENTRYPOINT ["python3", "-m", "polus.images.transforms.images.apply_flatfield"]
CMD ["--help"]
