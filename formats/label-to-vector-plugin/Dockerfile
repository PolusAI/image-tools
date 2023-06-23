FROM polusai/bfio:2.1.9

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"
ENV POLUS_LOG="INFO"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"

RUN mkdir /.cache && chmod 777 /.cache
RUN mkdir -m 777 -p ${EXEC_DIR}/numba_cache

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src

RUN pip3 install ${EXEC_DIR} --no-cache-dir

ENV NUMBA_CACHE_DIR=${EXEC_DIR}/numba_cache

ENTRYPOINT ["python3", "-m", "polus.plugins.formats.label_to_vector"]
CMD ["--help"]
