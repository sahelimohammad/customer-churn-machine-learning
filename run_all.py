"""Run EDA, supervised classification and unsupervised clustering in order."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
for item in ("eda/eda.py","classification/classification.py","clustering/clustering.py"):
    print("\nRunning",item,flush=True)
    subprocess.run([sys.executable,str(ROOT/item)],cwd=ROOT,check=True)
print("\nDone. See results/ and figures/.")
