"""Structured logging utilities"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlHandler(logging.Handler):
    """Handler that outputs structured logs in JSONL format"""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as JSON line"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_entry["exception"] = self.format(record)

        with open(self.filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def setup_logging(
    log_dir: Path,
    name: str = "nora",
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup structured logging for a run"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # JSONL file handler
    log_file = log_dir / "logs.jsonl"
    jsonl_handler = JsonlHandler(str(log_file))
    jsonl_handler.setLevel(level)
    logger.addHandler(jsonl_handler)

    return logger
