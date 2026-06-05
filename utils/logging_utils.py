"""
utils/logging_utils.py
-----------------------
Logging setup for the experiment.

Call setup_logging() once at the start of train.py.
Log messages go to both the console and a file under results/.
"""

from __future__ import annotations
from pathlib import Path
import logging
import sys


def setup_logging(log_file: Path, level: int = logging.INFO) -> None:
    """
    Configure root logger to write to both console and a file.

    Parameters
    ----------
    log_file : Path — destination log file (parent directories are created).
    level : int — logging level (default INFO).
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.info("Logging to %s", log_file)
