FROM tensorflow/tensorflow:2.15.0-gpu

# cat /etc/os-release
#
# NAME="Ubuntu"
# VERSION="20.04.5 LTS (Focal Fossa)"
# ID=ubuntu
# ID_LIKE=debian
# PRETTY_NAME="Ubuntu 20.04.5 LTS"
# VERSION_ID="20.04"
# HOME_URL="https://www.ubuntu.com/"
# SUPPORT_URL="https://help.ubuntu.com/"
# BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
# PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
# VERSION_CODENAME=focal
# UBUNTU_CODENAME=focal

# Install Python 3.12 from source (not available in deadsnakes PPA for Ubuntu 20.04)
RUN apt update && \
    apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev curl git && \
    cd /tmp && \
    wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz && \
    tar -xzf Python-3.12.7.tgz && \
    cd Python-3.12.7 && \
    ./configure --enable-optimizations --prefix=/usr/local && \
    make -j$(nproc) && \
    make altinstall && \
    cd / && \
    rm -rf /tmp/Python-3.12.7* && \
    apt remove -y build-essential wget && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install pip and create symbolic links
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    /usr/local/bin/python3.12 get-pip.py && \
    rm get-pip.py && \
    ln -sf /usr/local/bin/pip3.12 /usr/bin/pip3
RUN mkdir /.cache && chmod 777 /.cache

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"
ENV DATA_DIR="/data"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

# TODO: Change the tool_dir to the tool directory
ENV TOOL_DIR="regression/basic-flatfield-estimation-tool"

# Copy the repository into the container
RUN mkdir image-tools
COPY . ${EXEC_DIR}/image-tools

# Install the tool
WORKDIR ${EXEC_DIR}/image-tools/${TOOL_DIR}
RUN pip3 install . --no-cache-dir
WORKDIR ${EXEC_DIR}

# Set the entrypoint
# TODO: Change the entrypoint to the tool entrypoint
ENTRYPOINT ["python3", "-m", "polus.images.regression.basic_flatfield_estimation"]
CMD ["--help"]
