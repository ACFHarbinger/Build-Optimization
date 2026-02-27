"""File and data hashing utilities for change detection."""

import hashlib
import os
from typing import Optional


def hash_file(path: str, chunk_size: int = 65536) -> Optional[str]:
    """Compute SHA-256 hash of a file on disk.

    Args:
        path: Absolute or relative path to the file.
        chunk_size: Read chunk size in bytes (default 64 KB).

    Returns:
        Hex-encoded SHA-256 digest, or ``None`` if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of raw bytes.

    Args:
        data: Bytes to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()
