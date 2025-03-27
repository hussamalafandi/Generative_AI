import datetime
import logging
import os


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger with a file handler and console handler.

    Args:
        name (str): The name of the logger.
        log_file (str): The path to the log file.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger.
    """
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times in interactive environments.
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
