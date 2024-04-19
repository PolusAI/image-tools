FROM continuumio/miniconda3

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

ARG OMERO_USERNAME
ARG OMERO_PASSWORD
ENV OMERO_USERNAME=$OMERO_USERNAME
ENV OMERO_PASSWORD=$OMERO_PASSWORD


ARG conda_env=myenv

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

COPY pyproject.toml ${EXEC_DIR}
COPY environment.yml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src

RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/myenv/bin:$PATH
RUN /bin/bash -c "source activate myenv"

RUN pip3 install ${EXEC_DIR} --no-cache-dir

ENTRYPOINT ["python3", "-m", "polus.images.utils.omero_download"]
CMD ["--help"]
