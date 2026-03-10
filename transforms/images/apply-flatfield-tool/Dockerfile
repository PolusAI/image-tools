# bfiocpp (bfio dependency) has no wheel for arm64 + Python 3.13. Build with:
#   docker build --platform linux/amd64 -f transforms/images/apply-flatfield-tool/Dockerfile -t polusai/apply-flatfield-tool:2.0.2-dev0 .
FROM python:3.13-slim

ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"

WORKDIR ${EXEC_DIR}

# bfio/Bioformats needs a JVM (Debian Trixie has openjdk-21, not 17)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Build context = repo root. Copy only this tool.
COPY transforms/images/apply-flatfield-tool /opt/executables/app

RUN pip install --no-cache-dir /opt/executables/app

ENTRYPOINT ["python3", "-m", "polus.images.transforms.images.apply_flatfield"]
CMD ["--help"]
