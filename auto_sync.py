"""
Eataway Auto-Sync Script
========================
Workflow:
  1. Pull latest data from company MySQL database (local)
  2. Run feature engineering + model training locally
     (avoids server memory limits — training is CPU/RAM intensive).
  3. The training script exports the final predictions to Google Sheets.

Usage:
  python auto_sync.py

Scheduled runs: configure Windows Task Scheduler to run this script weekly.
"""

import sys
import time
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ============================================================
# Script logic (no need to modify below)
# ============================================================

FEATURE_SCRIPT = BASE_DIR / "feature.py"
TRAIN_SCRIPT   = BASE_DIR / "eataway_train_v7.py"


def run_local_training() -> bool:
    """Run feature engineering and model training locally."""
    python = sys.executable
    steps = [
        ("Feature Engineering", [python, str(FEATURE_SCRIPT)]),
        ("Model Training",      [python, str(TRAIN_SCRIPT)]),
    ]
    for name, cmd in steps:
        print(f"▶ Running locally: {name}...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(BASE_DIR), env=env,
        )
        for line in proc.stdout:
            print(f"  {line.rstrip()}")
        proc.wait()
        if proc.returncode != 0:
            print(f"  ✗ {name} failed (code={proc.returncode})")
            return False
        print(f"  ✓ {name} complete")
    return True


def main():
    print("=" * 55)
    print("  Eataway Auto-Sync  (local training mode)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Run local training pipeline (feature engineering + model training)
    # The training script handles exporting results to Google Sheets.
    if not run_local_training():
        print("✗ Local training failed — check errors above")
        sys.exit(1)

    print(f"\n✓ All done! Model trained and Google Sheet updated.")


if __name__ == "__main__":
    main()
