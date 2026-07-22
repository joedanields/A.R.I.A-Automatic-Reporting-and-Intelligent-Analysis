#!/usr/bin/env python3
"""
A.R.I.A. Setup Script
=====================
Creates virtual environment, installs dependencies, pulls Ollama models,
and verifies the environment is ready for development.

Usage:
    python scripts/setup.py
    python scripts/setup.py --skip-ollama   # skip Ollama model pulls
    python scripts/setup.py --backend-only  # only set up backend
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
VENV_DIR = BACKEND_DIR / "venv_new"


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and stream output."""
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT_DIR, check=False)
    if check and result.returncode != 0:
        print(f"  ERROR: command failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def create_venv() -> None:
    """Create the Python virtual environment."""
    if VENV_DIR.exists():
        print(f"  Virtual environment already exists at {VENV_DIR}")
    else:
        print("  Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])


def pip_install(requirements: Path) -> None:
    """Install packages from a requirements file."""
    pip = VENV_DIR / "Scripts" / "pip.exe"
    if not pip.exists():
        pip = VENV_DIR / "bin" / "pip"
    run([str(pip), "install", "-r", str(requirements)])


def pull_ollama_models() -> None:
    """Pull required Ollama models."""
    models = ["phi3:mini"]
    for model in models:
        print(f"  Pulling {model}...")
        result = run(["ollama", "pull", model], check=False)
        if result.returncode != 0:
            print(f"  WARNING: Failed to pull {model}. You can pull it later with: ollama pull {model}")


def verify() -> None:
    """Verify the installation."""
    python = VENV_DIR / "Scripts" / "python.exe"
    if not python.exists():
        python = VENV_DIR / "bin" / "python"

    print("\n  Verifying installation...")
    result = run(
        [str(python), "-c",
         "import fastapi; import langgraph; import chromadb; import faster_whisper; "
         "print('All core imports OK')"],
        check=False,
    )
    if result.returncode != 0:
        print("  ERROR: Some imports failed. Check the output above.")
        sys.exit(1)
    print("  Verification passed!")


def main() -> None:
    parser = argparse.ArgumentParser(description="A.R.I.A. Setup Script")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama model pulls")
    parser.add_argument("--backend-only", action="store_true", help="Only set up backend")
    args = parser.parse_args()

    print("=" * 60)
    print("A.R.I.A. Setup")
    print("=" * 60)

    # Backend
    print("\n[1/4] Creating virtual environment...")
    create_venv()

    print("\n[2/4] Installing backend dependencies...")
    pip_install(BACKEND_DIR / "requirements.txt")

    if not args.backend_only:
        # Frontend
        print("\n[3/4] Installing frontend dependencies...")
        frontend_dir = ROOT_DIR / "frontend"
        if (frontend_dir / "package.json").exists():
            run(["npm", "install"], cwd=frontend_dir, check=False)
        else:
            print("  Skipping frontend (no package.json found)")
    else:
        print("\n[3/4] Skipping frontend (--backend-only)")

    # Ollama
    if not args.skip_ollama:
        print("\n[4/4] Pulling Ollama models...")
        pull_ollama_models()
    else:
        print("\n[4/4] Skipping Ollama models (--skip-ollama)")

    # Verify
    verify()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"  Backend:  cd backend && python main.py")
    print(f"  Frontend: cd frontend && npm run dev")
    print("=" * 60)


if __name__ == "__main__":
    main()
