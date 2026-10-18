#!/usr/bin/env bash
set -euo pipefail
case "${1:---verify}" in
 --download) python -m src.data --download ;;
 --train) python -m src.train --aco; python -m src.forward --verify; python -m src.plotting; python -m src.build_site ;;
 --train-fast) python -m src.train; python -m src.forward --verify; python -m src.plotting; python -m src.build_site ;;
 --verify) python -m src.forward --verify; python -m src.plotting; python -m src.build_site ;;
 --site) python -m src.build_site ;;
 --test) python -m unittest discover -s tests -v ;;
 *) echo 'Usage: bash run_all.sh [--download|--train|--train-fast|--verify|--site|--test]' >&2; exit 2 ;;
esac
