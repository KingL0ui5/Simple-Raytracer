"""
This module contains the optical elements of the system.
L Liu 22/05
"""
import numpy as np
from raytracer.rays import Ray

class OpticalElement:
    def intercept(self, ray):
            raise NotImplementedError('intercept() needs to be implemented in derived classes')
    def propagate_ray(self, ray):
            raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')
            
class SphericalRefraction(OpticalElement):
    def __init__(self, z_0: int, aperture: int, curvature: float, n_1: float, n_2: float):
        self.__z_0 = z_0
        self.__aperture = aperture
        self.__curvature = curvature
        self.__n_1 = n_1
        self.__n_2 = n_2
        
    def z_0(self):
        return self.__z_0
    
    def aperture(self):
        return self.__aperture  
        
    def curvature(self):
        return self.__curvature
    
    def n_1(self):
        return self.__n_1
    
    def n_2(self):  
        return self.__n_2
    
    def intercept(self, ray):
        R = 1. / self.__curvature
        r = ray.pos() - np.array([0., 0., self.__z_0 + R]) 
        b = np.dot(ray.direc(), r)
        
        disc = b**2 - (np.dot(r, r) - R**2)
        if disc < 0:
            # No intersection
            return None

        d = [0., 0.]
        sqrt_disc = np.sqrt(disc)
        d[0] = -b + sqrt_disc
        d[1] = -b - sqrt_disc
        
        d.sort()        
        # choosing the right intersection
        if d[0] < 1e-8:
            d[0] = d[1]
            if d[1] < 1e-8:   
                return None
                
        d = d[0]
            
        intercept = ray.pos() + d * ray.direc()
        
        if intercept[0]**2 + intercept[1]**2 > (self.__aperture)**2:
            return None
        
        return intercept
        