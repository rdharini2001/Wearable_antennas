# Computational and electromagnetic methods

WEAVE combines antenna response data with Gaussian-kernel support vector regression and continuous-domain ant-colony parameter selection. Its regression experiments use operating frequency, radiating-patch geometry and slot geometry to model signal strength. The training study includes the twelve regression families described in the research article and a feature-ablation experiment in which operating frequency is omitted.

Dense frequency sweeps provide additional information about the local shape of resonant antenna responses. A training-selected interpolation term can therefore complement the global regression model when frequency-response observations from the same geometry are available. A geometry-conditioned distributional estimator separately describes the response variation across the sampled operating band.

The electromagnetic design module uses standard microstrip relations for patch width, effective permittivity, fringing extension, patch length and ground-plane dimensions. Textile material behavior is illustrated using the cotton, polyester and poly-amide substrate conditions in the antenna study.

**Complete derivations and experimental details:** [Supplementary Information (PDF)](SUPPLEMENTARY_METHODS.pdf) · [Readable source](SUPPLEMENTARY_METHODS.md)
