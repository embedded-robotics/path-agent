import logging
import sys

_DEF_FMT = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"

def get_logger(name: str = "pathrag", level: int = logging.INFO, fmt: str = _DEF_FMT) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
