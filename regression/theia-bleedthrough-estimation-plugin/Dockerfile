FROM tensorflow/tensorflow:2.12.0-gpu

# Output from `cat /etc/os-release` in the base container
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

# Instal Python 3.9
RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.9 python3.9-distutils curl git maven openjdk-8-jre -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Symbolic link to python3.9
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"
ENV POLUS_LOG="INFO"
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"

RUN mkdir /.cache && chmod 777 /.cache

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

# Copy the python package
COPY pyproject.toml ${EXEC_DIR}
COPY VERSION ${EXEC_DIR}
COPY README.md ${EXEC_DIR}
COPY src ${EXEC_DIR}/src

# Install the python package
RUN pip3 install ${EXEC_DIR} --no-cache-dir

ENTRYPOINT ["python3", "-m", "polus.plugins.regression.theia_bleedthrough_estimation"]
CMD ["--help"]
