FROM vector-label-setup

# Default command. Additional arguments are provided through the command line
ENTRYPOINT ["python3", "/opt/executables/label_to_vector.py"]