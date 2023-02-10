FROM polusai/bfio:2.1.9

ENV POLUS_EXT=".ome.tif"
ENV POLUS_LOG="INFO"
ENV DATA_DIR="/data"
ENV EXEC_DIR="/opt/executables"

RUN pip3 install --upgrade pip setuptools wheel

WORKDIR ${EXEC_DIR}

RUN mkdir roi_relabel && \
    touch roi_relabel/__init__.py

COPY pyproject.toml ${EXEC_DIR}/pyproject.toml

RUN pip3 install -e . --no-cache-dir

COPY roi_relabel roi_relabel

ENTRYPOINT ["python3", "-m", "roi_relabel"]
