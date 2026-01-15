#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

pause_with_message() {
  echo
  echo "$1"
  read -rp "Press Enter to close..." _
}

if ! command -v python3 >/dev/null 2>&1; then
  pause_with_message "Python 3 is required but was not found. Please install Python 3.x and try again."
  exit 1
fi

trap 'pause_with_message "Something went wrong. Try deleting the .venv folder and double-clicking again."' ERR

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment in .venv..."
  /usr/bin/env python3 -m venv .venv
fi

source ".venv/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
python -m pip install -r requirements.txt

echo "Starting app..."
python -m streamlit run app_framework.py
