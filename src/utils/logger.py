import logging

def setup_logger(name: str):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(mesaage)s"
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger