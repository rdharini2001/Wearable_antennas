"""Closed-form rectangular microstrip design relations used in the GCITC study."""
from __future__ import annotations
import math

C0 = 299_792_458.0


def patch_dimensions(freq_ghz: float, relative_permittivity: float, height_mm: float) -> dict:
    """Calculate patch and ground dimensions from the article's equations (2–7).

    The transmission-line approximation is for an un-slotted rectangular patch;
    it does not solve S11, realized gain, near-body coupling or fringing from slots.
    """
    f = float(freq_ghz) * 1e9
    eps = float(relative_permittivity)
    h = float(height_mm) * 1e-3
    if not (math.isfinite(f) and math.isfinite(eps) and math.isfinite(h)):
        raise ValueError('Frequency, dielectric constant and height must be finite')
    if f <= 0 or eps < 1 or h <= 0:
        raise ValueError('Expected positive frequency/height and relative permittivity >= 1')
    W = C0/(2*f)*math.sqrt(2/(eps+1))
    ratio = W/h
    eps_eff = (eps+1)/2 + (eps-1)/2*(1+12/ratio)**(-0.5)
    delta_length = h*0.412 * ((eps_eff+0.3)*(ratio+0.264))/((eps_eff-0.258)*(ratio+0.8))
    length = C0/(2*f*math.sqrt(eps_eff)) - 2*delta_length
    if length <= 0:raise ValueError('Substrate height is incompatible with a positive patch length')
    return dict(patch_width_mm=W*1e3, patch_length_mm=length*1e3,
                effective_permittivity=eps_eff, fringing_extension_mm=delta_length*1e3,
                ground_width_mm=(W+6*h)*1e3, ground_length_mm=(length+6*h)*1e3,
                freq_ghz=freq_ghz, relative_permittivity=eps, height_mm=height_mm)
