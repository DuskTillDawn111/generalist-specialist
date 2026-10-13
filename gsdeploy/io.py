"""Strict JSON output, published atomically without overwriting experiment data."""

import json
import os
from pathlib import Path
import tempfile


def native(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dumps(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=native)
        + "\n"
    )


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(dumps(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
