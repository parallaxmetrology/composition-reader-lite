#!/bin/sh
# Create a virtual environment on first run, then serve the app locally.
set -eu
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -e .
  PYTHONPATH="$PWD/src:$PWD/vendor" .venv/bin/python -m composition_reader.cli install-model
fi
export PYTHONPATH="$PWD/src:$PWD/vendor${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m composition_reader.cli serve --open "$@"
