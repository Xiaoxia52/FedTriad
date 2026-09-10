import json
import os
import time
import warnings
from pathlib import Path


def replace_with_retry(temp, path, attempts=8):
    """Keep atomic replacement; retry transient Windows sharing/access errors.

    Permanent failure remains fatal and leaves the previous destination and
    temporary file intact. Never unlink the destination to bypass a lock.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            os.replace(str(temp), str(path))
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            if attempt == 0:
                warnings.warn("File replacement temporarily denied; retrying without deleting: " + str(path))
            time.sleep(min(0.1 * 2 ** attempt, 1.0))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temp, path)
