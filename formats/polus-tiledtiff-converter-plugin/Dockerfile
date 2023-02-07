FROM maven:3-jdk-8
ARG EXEC_DIR="/opt/executables"
RUN mkdir -p ${EXEC_DIR}
COPY . ${EXEC_DIR}/.
WORKDIR ${EXEC_DIR}
RUN ./mvn-packager.sh

FROM openjdk:8-alpine
COPY VERSION /
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs
COPY --from=0 /opt/executables/target/tiledtiff-converter-plugin-1.1.1-jar-with-dependencies.jar ${EXEC_DIR}/.
WORKDIR ${EXEC_DIR}
ENTRYPOINT ["java", "-jar", "tiledtiff-converter-plugin-1.1.1-jar-with-dependencies.jar"]
