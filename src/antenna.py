"""Classical rectangular microstrip patch equations used in the manuscript."""
from __future__ import annotations
import argparse,json,math
C0=299792458.0

def dimensions(frequency_ghz=2.4, epsilon_r=1.1, height_mm=1.6):
    if frequency_ghz <=0 or epsilon_r<=1 or height_mm<=0:
        raise ValueError('Frequency, dielectric constant >1 and substrate height must be positive')
    f=frequency_ghz*1e9;h=height_mm
    w_mm=C0/(2*f)*math.sqrt(2/(epsilon_r+1))*1e3
    eff=(epsilon_r+1)/2+(epsilon_r-1)/2/(1+12*h/w_mm)**.5
    delta_l=.412*h*((eff+.3)*(w_mm/h+.264))/((eff-.258)*(w_mm/h+.8))
    l_mm=C0/(2*f*math.sqrt(eff))*1e3-2*delta_l
    return dict(frequency_ghz=frequency_ghz,relative_permittivity=epsilon_r,substrate_height_mm=h,
                effective_permittivity=eff,fringing_extension_mm=delta_l,
                patch_width_mm=w_mm,patch_length_mm=l_mm,
                ground_width_mm=w_mm+6*h,ground_length_mm=l_mm+6*h)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ghz',type=float,default=2.4);p.add_argument('--epsilon',type=float,default=1.1);p.add_argument('--height-mm',type=float,default=1.6)
    a=p.parse_args();print(json.dumps(dimensions(a.ghz,a.epsilon,a.height_mm),indent=2))
