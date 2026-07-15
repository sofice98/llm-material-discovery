"""Console-output logging helpers for command-line scripts."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class TeeStream:
    """Mirror a text stream to the console and a log file."""

    def __init__(self, console: Any, log_file: Any) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.console.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.console.isatty()


def configure_script_logging(script_path: str | Path) -> Path:
    """Mirror stdout and stderr to ``logs/<script name>.log`` beside a script."""
    script = Path(script_path).resolve()
    log_path = script.parent / "logs" / f"{script.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n{'=' * 24} Started {timestamp} {'=' * 24}\n")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_path
