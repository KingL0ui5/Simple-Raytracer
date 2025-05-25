"""
A module to investigate generators.
Louis Liu 15/05/2025
"""
import numpy as np

def rtrings(rmax=5., nrings=5, multi=6):
    yield 0.0, 0.0 
    for ring in range(1, nrings + 1):
        r = (rmax/nrings) * ring
        npoints = ring * multi
        
        for point in range(npoints):
            theta = ((np.pi * 2)/npoints) * point
            yield [r,theta]
        