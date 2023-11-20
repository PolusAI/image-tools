FROM r-base:4.0.3

	
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN apt-get update && apt-get install -y python3

#Create folders
RUN mkdir -p /{EXEC_DIR} \
    && mkdir -p /{DATA_DIR}/csv \
    && mkdir -p /{DATA_DIR}/outputs

#Copy executable
COPY src /{EXEC_DIR}/

RUN Rscript /{EXEC_DIR}/Requirements.R
WORKDIR /{EXEC_DIR}

ENTRYPOINT ["Rscript", "/opt/executables/main.R"]