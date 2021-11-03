FROM tensorflow/tensorflow:2.4.1-gpu


COPY VERSION ${EXEC_DIR}
COPY src ${EXEC_DIR}/


RUN python3 -m pip install --upgrade pip setuptools wheel 


RUN python3 -m pip install -r ${EXEC_DIR}/requirements.txt
                                                                                                                                                                                            

ENTRYPOINT ["python3", "main.py"]