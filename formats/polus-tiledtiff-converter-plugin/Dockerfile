FROM openjdk:8-alpine
COPY VERSION /
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs
COPY ./target/polus-tiledtiff-converter-plugin-1.1.1-jar-with-dependencies.jar ${EXEC_DIR}/.
WORKDIR ${EXEC_DIR}
ENTRYPOINT ["java", "-jar", "polus-tiledtiff-converter-plugin-1.1.1-jar-with-dependencies.jar"]
