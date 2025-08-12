import logging
import sys
from pathlib import Path
from . import config as cfg


def setup_logging() -> logging.Logger:
    cfg.ensure_dirs()
    log_path = cfg.LOG_DIR / "training5.log"
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format=cfg.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("training5")

