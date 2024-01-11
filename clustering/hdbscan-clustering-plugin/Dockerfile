FROM polusai/bfio:2.3.3

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".arrow"


# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}

RUN apt --no-install-recommends -y autoremove --purge python3.9-minimal python3.9\
    && apt-get update && apt-get install --no-install-recommends -y build-essential python3.9-dev

RUN pip3 install ${EXEC_DIR} --no-cache


ENTRYPOINT ["python3", "-m", "polus.plugins.clustering.hdbscan_clustering"]
CMD ["--help"]
