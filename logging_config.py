# logging_config.py
import logging
import sys
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": "app.log",
            "mode": "a",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
    "loggers": {
        "uvicorn": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        "fastapi": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
    },
}

def setup_logging():
    dictConfig(LOGGING_CONFIG)
