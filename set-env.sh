#!/bin/bash

PROJ_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

export PYTHONPATH=$PROJ_DIR:$PYTHONPATH
