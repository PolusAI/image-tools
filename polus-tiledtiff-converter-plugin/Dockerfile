FROM openjdk:8-alpine

COPY VERSION /

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

#Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

#Copy executable
COPY target/polus-tiledtiff-converter-plugin-1.0.0-jar-with-dependencies.jar ${EXEC_DIR}/.

WORKDIR ${EXEC_DIR}

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["java", "-jar", "polus-tiledtiff-converter-plugin-0.0.1-SNAPSHOT-jar-with-dependencies.jar"]