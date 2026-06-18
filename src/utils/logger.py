"""
Provide shared logger setup for project modules.

This utility creates console loggers with a consistent format so runtime
scripts and services can emit comparable operational messages.
"""

import logging

def setup_logger(name: str):
    """
    Create or reuse a named console logger.

    Args:
        name: Logger name, usually the module name.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Avoid duplicate handlers when setup is called multiple times.
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(mesaage)s"
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
