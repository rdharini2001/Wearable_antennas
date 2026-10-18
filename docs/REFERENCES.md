# References and methodological background

1. **Manuscript.** Raghavan, D., Sujatha, B., Chinmayi, B. S., Rao, B. B., and Kiran, J. (2023). *Ant Colony Optimization Based Support Vector Regression Algorithm for Performance Prediction of Wearable Textile Antennas.* IEEE GCITC. [doi:10.1109/GCITC60406.2023.10425975](https://doi.org/10.1109/GCITC60406.2023.10425975). Source for the 12-model numerical comparisons, geometry study, optimization constants, and HFSS results.

2. **Ant-colony search.** Dorigo, M., and Blum, C. (2005). Ant colony optimization theory: a survey. *Theoretical Computer Science*, 344(2–3), 243–278. [doi:10.1016/j.tcs.2005.05.020](https://doi.org/10.1016/j.tcs.2005.05.020). Mathematical motivation for pheromone-based candidate sampling and reinforcement.

3. **Support vector regression.** Smola, A. J., and Schölkopf, B. (2004). A tutorial on support vector regression. *Statistics and Computing*, 14, 199–222. [doi:10.1023/B:STCO.0000035301.49549.88](https://doi.org/10.1023/B:STCO.0000035301.49549.88). SVR loss, kernel and regularization formulation.

4. **Model discrepancy.** Kennedy, M. C., and O'Hagan, A. (2001). Bayesian calibration of computer models. *Journal of the Royal Statistical Society: Series B*, 63(3), 425–464. [doi:10.1111/1467-9868.00294](https://doi.org/10.1111/1467-9868.00294). Background for treating residual correction separately from the primary predictive model.

5. **Calibration estimators.** Deville, J.-C., and Särndal, C.-E. (1992). Calibration estimators in survey sampling. *Journal of the American Statistical Association*, 87(418), 376–382. [doi:10.1080/01621459.1992.10475217](https://doi.org/10.1080/01621459.1992.10475217). Mathematical context for nonnegative weighted estimators subject to numerical constraints. The evaluation operators in WEAVE are specifically defined in its computational methods.

6. **Signal-data source.** [MHassaanButt/Antenna-design-using-ML](https://github.com/MHassaanButt/Antenna-design-using-ML), cited as reference [16] in the GCITC paper. Data columns: `TestFreq`, `PatchLength`, `PatchWidth`, `SlotLength`, `SlotWidth`, `Strength`. The release includes the 1,266 publicly available numerical observations.
