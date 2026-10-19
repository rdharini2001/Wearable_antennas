# Publish WEAVE V2

The V2 release is a drop-in overlay for the [Wearable_antennas repository](https://github.com/rdharini2001/Wearable_antennas). The overlay is intended to be applied to a clean checkout and preserves the original dataset's full response precision.

```bash
git clone https://github.com/rdharini2001/Wearable_antennas.git
cd Wearable_antennas
# Extract WEAVE-V2-GitHub-overlay.zip over the checkout.
# The overlay deliberately excludes data/comm_antenna.csv.
python -m pip install -r requirements.txt
bash run_all.sh --full
```

Inspect the newly generated `results/enhanced/`, `results/transport/`, `docs/downloads/`, and `docs/index.html` before committing. The independent ACO search, baseline training, distribution fitting, scorecard, exported model replay, website build, and tests are run by that one command. A second validation pass without reselection is available using `bash run_all.sh --evaluate`.

```bash
git add README.md REFERENCES.bib requirements.txt run_all.sh src tests data docs results .github
# Review the diff, then commit and push.
```

For GitHub Pages choose **Settings → Pages → Deploy from a branch → main → /docs**. The site is fully static; `docs/index.html` contains its own serialized experiment data, and CSV downloads are shipped alongside it.

## Interpretation of the release

The 2023 paper is the initial study. V2 adds higher-resolution spectral-response interpolation and frequency-marginal uncertainty modeling. It independently exceeds the 48 conventional pointwise reference thresholds in the frequency-aware arm. The combined 96/96 **numerical** threshold result includes 48 other measurements using Wasserstein-based distributional evaluation and must not be described as 96 exact or identical conventional metrics. The full-spectrum geometry-only pointwise scorecard remains 41/48; the reported HFSS values are inherited from the paper.

Do not change a reported metric, split, benchmark definition, or source table during release preparation merely to improve numerical agreement. Regenerate all artifacts after changing the dataset, software versions, or source code.
