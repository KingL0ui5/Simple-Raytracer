"""
A module to explore lenses in the raytracer setup 
L Liu 25/05/2025
"""

class PlanoConvex:
    def __init__(self, z_0: int, curvature1: float, curvature2: float, n_inside: float, n_outside: float, thickness: float, aperture: float):
        self.__z_0 = z_0
        self.__curvature1 = curvature1
        self.__curvature2 = curvature2
        self.__n_inside = n_inside
        self.__n_outside = n_outside
        self.__thickness = thickness
        self.__aperture = aperture
    
    