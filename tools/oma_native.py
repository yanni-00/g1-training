"""Call the installed OMA CLI through the project-readable Python process."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cli = root.parents[1] / 'oh-my-algorithm/bin/oma.js'
raise SystemExit(subprocess.call(['node', str(cli), *sys.argv[1:]], cwd=root))
