FROM polusai/bfio:2.1.9

# environment variables defined in polusai/bfio
# environment variables
ENV POLUS_LOG_LEVEL="DEBUG"
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

# TOOL_DIR Argument to the Dockerfile
ARG TOOL_DIR="."

RUN apt-get -y update
RUN apt-get -y install git

# Copy the repository into the container
RUN mkdir rt_cetsa
COPY rt-cetsa-intensity-extraction-tool ${EXEC_DIR}/rt_cetsa/features/rt-cetsa-intensity-extraction-tool
COPY rt-cetsa-plate-extraction-tool ${EXEC_DIR}/rt_cetsa/segmentation/rt-cetsa-plate-extraction-tool

# Install the tool
RUN pip3 install "${EXEC_DIR}/rt_cetsa/features/rt-cetsa-intensity-extraction-tool"

ENTRYPOINT ["python3", "-m", "polus.images.features.rt_cetsa_intensity_extraction"]
CMD ["--help"]
