#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:---full}" in
  --full)
    python -m src.discovery --iterations "${ACO_ITERATIONS:-240}" --ants "${ACO_ANTS:-21}" --max-evaluations "${ACO_EVALUATIONS:-65}" --folds "${ACO_FOLDS:-3}"
    ;;
  --evaluate)
    test -f results/discovery/results.json || { echo 'Run training before evaluation.' >&2; exit 2; }
    ;;
  --site)
    python -m src.build_publication_site
    exit 0
    ;;
  *) echo 'Usage: bash run_all.sh [--full|--evaluate|--site]' >&2; exit 2;;
esac
python -m src.enhanced
python -m src.marginal
python -m src.spectral_validation
python -m src.scorecard
python -m src.transport
python -m src.audit
python -m src.benchmark --repeats 200
python -m src.build_publication_site
python -m unittest discover -s tests -v
