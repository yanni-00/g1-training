"""Restore the byte-exact training assets from the GitHub branch archive."""
import json
import shutil
import tarfile
from pathlib import Path

from .contract import ROOT, sha256


def ensure_assets(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "configs/source_manifest.json").read_text(encoding="utf-8"))
    if all((root / row["destination"]).is_file() for row in manifest["assets"]):
        return
    archive = root / "resources/f1_assets.tar.gz"
    description = json.loads((root / "configs/assets_archive.json").read_text(encoding="utf-8"))
    if sha256(archive) != description["sha256"]:
        raise ValueError("Training asset archive hash mismatch")
    target_root = (root / "resources").resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            target = (target_root / member.name).resolve()
            if not target.is_relative_to(target_root) or not member.isfile():
                raise ValueError("Unexpected archive entry")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    for row in manifest["assets"]:
        if sha256(root / row["destination"]) != row.get("training_sha256", row["sha256"]):
            raise ValueError("Restored F1 asset differs from registered training assets")
