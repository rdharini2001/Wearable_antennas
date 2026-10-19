# Publishing the WEAVE research companion

The repository contains the paper's computational models, numerical experiments, supplementary information and GitHub Pages research atlas.

## Use the standalone release

```bash
python -m pip install -r requirements.txt
bash run_all.sh --full
```

The training command fits the regressors, calibrates the frequency-response and distributional estimators, regenerates the results, and builds the research website in `docs/`.

For a website-only refresh from saved experimental results:

```bash
bash run_all.sh --site
```

## Update the existing GitHub project

Create a branch of `rdharini2001/Wearable_antennas` and retain the repository's original `data/comm_antenna.csv`, which contains higher-precision response measurements. Copy the **contents** of this release over the existing project, deleting files from the old version that are not included here. Restore the original antenna CSV and run the full experiment before committing the release. This ensures that the fitted models, evaluation tables and website are derived from the precise public source data.

The static site is designed for **GitHub Pages → Deploy from a branch → main → /docs**. Its primary entry point is `docs/index.html`; the mathematical supplement is `docs/SUPPLEMENTARY_METHODS.pdf`.

The Markdown manuscript for the supplement is supplied as `docs/SUPPLEMENTARY_METHODS.md`, with its typesetting preamble under `docs/source/`. The source article is linked through its publisher DOI.

The website and supplement can be updated independently of the regression model when only editorial changes are needed. Numerical revisions should be regenerated from the training workflow so that the site and exported results remain synchronized.
