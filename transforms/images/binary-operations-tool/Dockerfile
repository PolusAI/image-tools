# Build from repo root (monorepo) or from this tool directory — both work.
FROM polusai/bfio:2.5.0
ENV EXEC_DIR="/opt/executables" POLUS_IMG_EXT=".ome.tif" POLUS_TAB_EXT=".csv" POLUS_LOG="INFO"
WORKDIR ${EXEC_DIR}
ENV TOOL_DIR="transforms/images/binary-operations-tool"
RUN mkdir -p image-tools
COPY . ${EXEC_DIR}/image-tools
RUN pip3 install -U pip setuptools wheel \
  && python3 -c 'import sys; assert sys.version_info>=(3,11)' \
  && R="${EXEC_DIR}/image-tools" && M="$R/$TOOL_DIR" \
  && if [ -f "$M/pyproject.toml" ]; then pip3 install --no-cache-dir "$M"; \
     else pip3 install --no-cache-dir "$R"; fi
ENTRYPOINT ["python3", "-m", "polus.images.transforms.images.binary_operations"]
CMD ["--help"]
