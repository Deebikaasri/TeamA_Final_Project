import logging

def setup_logger(name: str, level: str = "INFO"):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
