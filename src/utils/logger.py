import logging
import os

# Define log file path
log_file_path = 'data/logs/BenchmarkLogging.log'  # Update this with your desired path

def setup_logger():
    logger = logging.getLogger('MyAppLogger')
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Check if handler already exists to avoid duplicate logs
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == fh.baseFilename for h in logger.handlers):
        logger.addHandler(fh)

    return logger
