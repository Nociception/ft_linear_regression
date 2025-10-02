#!/bin/bash
VENV=.venv
PY=$VENV/bin/python
PIP=$VENV/bin/pip

if [ ! -d "$VENV" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m pip install --user --upgrade virtualenv
    python3 -m virtualenv $VENV
    $PIP install --upgrade pip
    $PIP install -r requirements.txt
    echo "Setup complete!"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 {train|predict} [args...]"
    exit 1
fi

# Uncomment the following line to check the python files are run in the proper environment (the virtal one)
# echo "Using Python from venv: $($PY -c 'import sys; print(sys.executable)')"

cmd="$1"; shift

case "$cmd" in
    train)   $PY train.py "$@" ;;
    predict) $PY predict.py "$@" ;;
    *)
        echo "Unknown command '$cmd'. Use train or predict."
        exit 1
        ;;
esac
