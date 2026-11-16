"""Restore the pinned, lossless training assets before importing Isaac Lab."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def ensure_assets(root):
    root = Path(root)
    manifest = json.loads((root.parent/'.oma/impl/source_manifest.json').read_text(encoding='utf-8'))
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    if all((root/item['path']).is_file() and sha(root/item['path']) == item['sha256'] for item in manifest['assets']):
        return
    description = json.loads((root/'configs/assets_archive.json').read_text(encoding='utf-8'))
    archive = root/description['path']
    if sha(archive) != description['sha256']:
        raise ValueError('F1 asset archive identity mismatch')
    destination = (root/'resources').resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        for item in tar:
            target = (destination/item.name).resolve()
            if not target.is_relative_to(destination) or not item.isfile():
                raise ValueError('Non-file or escaping asset archive member')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    for item in manifest['assets']:
        if sha(root/item['path']) != item['sha256']:
            raise ValueError('Restored asset is not the registered F1 training asset')
