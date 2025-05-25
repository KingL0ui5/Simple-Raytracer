"""
This module contains the optical elements of the system.
L Liu 22/05
"""
import numpy as np
from .rays import Ray
from .physics import refract 

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
    
    def propagate_ray(self, ray): # test 
        intercept = self.intercept(ray)
        if intercept is None:
            return None
        
        if self.__curvature == 0:
            normal = np.array([0., 0., 1.])
        else:
            R = 1. / self.__curvature
            normal = intercept - np.array([0., 0., self.__z_0 + R])
            normal /= np.linalg.norm(normal)
        
        refracted_direc = refract(ray.direc(), normal, self.__n_1, self.__n_2)
        if refracted_direc is None:
            return None
        
        ray.append(intercept, refracted_direc)     
        
    def focal_point(self) -> float:
        R = 1. / self.__curvature
        z = self.__z_0 + (self.__n_2 * R) / (self.__n_2 - self.__n_1) # lensmakers formula
        return z
        
    def plot_surface(self, ax, resolution=100):
        r_max = self.__aperture / 2
        curvature = self.__curvature

        x = np.linspace(-r_max, r_max, resolution)
        y = np.linspace(-r_max, r_max, resolution)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        mask = R2 <= r_max**2

        Z = np.zeros_like(X) + self.__z_0
        if curvature != 0:
            R = 1 / curvature
            Z[mask] += R - np.sqrt(R**2 - R2[mask])
        else:
            Z[mask] += 0  

        Z[~mask] = np.nan

        ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan', rstride=1, cstride=1, linewidth=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        

class OutputPlane(OpticalElement):
    def __init__(self, z_0: float):
        self.__z_0 = z_0
        
    def z_0(self):
        return self.__z_0
    
    def intercept(self, ray):
        if ray.direc()[2] == 0: 
            return None
        k = (self.__z_0 - ray.pos()[2]) / ray.direc()[2]
        intercept = ray.pos() + k * ray.direc()
        return intercept
    
    def propagate_ray(self, ray):
        intercept = self.intercept(ray)
        ray.append(intercept, ray.direc()) # unchanged direction