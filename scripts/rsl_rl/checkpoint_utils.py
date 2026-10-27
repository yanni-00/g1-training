"""Checkpoint provenance checks that do not require Isaac Lab."""

import hashlib
import re


def verify_checkpoint_sha256(path, expected):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Checkpoint SHA-256 must contain exactly 64 hexadecimal characters")
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError(f"Checkpoint SHA-256 mismatch for {path}: expected {expected.lower()}, got {actual}")
    return actual
