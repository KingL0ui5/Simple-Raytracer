"""
A module to investigate generators.
Louis Liu 15/05/2025
"""
import numpy as np

def rtrings(rmax=5., nrings=5, multi=6):
    """
    Generates concentric rings of points in 2D polar coordinates.

    Args:
        rmax (float, optional): Maximum radius of rings. Defaults to 5
        nrings (int, optional): Number of rings to the maximum. Defaults to 5.
        multi (int, optional): Number of points in each ring. Defaults to 6.

    Yields:
        array: coordinates in polar form.
    """
    yield 0.0, 0.0 
    for ring in range(1, nrings + 1):
        r = (rmax/nrings) * ring
        npoints = ring * multi
        
        for point in range(npoints):
            theta = ((np.pi * 2)/npoints) * point
            yield [r,theta]
        