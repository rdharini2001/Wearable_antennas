# Antenna observation data

`comm_antenna.csv` contains 1,266 antenna frequency-response rows arranged in eight geometry sweeps. It is a six-decimal export derived from the source observations used in the independent WEAVE study. Antenna dimensions and test frequencies retain their source precision; response values are rounded to six decimal places.

For publication in the original [Wearable_antennas repository](https://github.com/rdharini2001/Wearable_antennas), retain its higher-precision `data/comm_antenna.csv` and rerun the experiment to regenerate consistent artifacts.

The input observations originate from [MHassaanButt/Antenna-design-using-ML](https://github.com/MHassaanButt/Antenna-design-using-ML), cited as reference [16] by the initial GCITC paper.

`manuscript_results.csv` preserves the original paper's Tables I and II as post-training comparison references. `substrate_simulation.csv` and `aco_hfss_comparison.csv` retain cited simulation values. The benchmark tables are not accessed by training, model selection or early stopping.
