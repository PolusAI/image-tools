FROM polusai/bfio:2.1.9

# environment variables defined in polusai/bfio
# ENV EXEC_DIR="/opt/executables"
# ENV DATA_DIR="/data"
# ENV POLUS_EXT=".ome.tif"
# ENV POLUS_LOG="INFO" # Change to WARNING for fewer logs, and DEBUG for debugging
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"

# Work directory defined in the base container
# WORKDIR ${EXEC_DIR}

COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src

RUN pip3 install ${EXEC_DIR} --no-cache-dir

ENTRYPOINT ["python3", "-m", "polus.images.transforms.images.roi_relabel"]
CMD ["--help"]
