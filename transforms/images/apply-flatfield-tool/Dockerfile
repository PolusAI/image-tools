FROM python:3.13-slim

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}


ENV TOOL_DIR="transforms/images/apply-flatfield-tool"

# Copy the repository into the container
RUN mkdir image-tools
COPY . ${EXEC_DIR}/image-tools

# Install the tool
RUN pip3 install "${EXEC_DIR}/image-tools/${TOOL_DIR}" --no-cache-dir


ENTRYPOINT ["python3", "-m", "polus.images.transforms.images.apply_flatfield"]
CMD ["--help"]
