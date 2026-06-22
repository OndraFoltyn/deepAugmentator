import logging
import os
from logging import Logger


def create_logger(log_file_path: str = None, log_level: int = logging.INFO) -> Logger:
    """Create and return a logger that writes to the console and optionally to a file."""
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # File handler if log_file_path is provided
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger