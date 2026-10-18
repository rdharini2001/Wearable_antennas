#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:---verify}" in
 --train)
    python -m src.aco
    python -m src.train
    python -m src.forward
    python -m src.verify
    python -m src.build_site
    python -m unittest discover -s tests -v
    ;;
 --verify)
    python -m src.forward
    python -m src.verify
    python -m src.build_site
    python -m unittest discover -s tests -v
    ;;
 --aco)
    python -m src.aco
    ;;
 *) echo 'Usage: bash run_all.sh [--train|--verify|--aco]' >&2; exit 2;;
esac
