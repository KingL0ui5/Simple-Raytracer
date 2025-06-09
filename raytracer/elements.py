"""
This module contains the optical elements of the system.
L Liu 22/05
"""
import numpy as np
from .rays import Ray
from .physics import refract 
import math

class OpticalElement:
    """
    Represents Optical Elements used in the system.
    """
    def intercept(self, ray):
        # must be implemented in all derived classes
            raise NotImplementedError('intercept() needs to be implemented in derived classes')
        
    def propagate_ray(self, ray):
            raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')
            
class SphericalRefraction(OpticalElement):
    """
    A spherically refracting optical element (inheriting from OpticalElement)
    """
    def __new__(cls, z_0: float, curvature: float, n_1: float, n_2: float, aperture: float):
        """
        Creates a new instance of the refractor depending on the curvature value. If the curvature is zero, a 
        PlaneRefractor is created.

        Args:
            z_0 (float): position of surface along the optical (z-axis).
            curvature (float): the curvature of the surface - for spherical refractor, curvature = 1/R 
                where R is the radius of curvature.
            n_1 (float): Refractive index of media before surface.
            n_2 (float): Refractive index of media behind surface.
            aperture (float): The circular aperture of the surface.

        Returns:
            Creates either a PlaneRefraction surface or calls constructor.
        """
        if curvature == 0:
            return PlaneRefraction(z_0 = z_0, n_1 = n_1, n_2 = n_2, aperture = aperture)
        return super().__new__(cls)
    
    def __init__(self, z_0: float, aperture: float, curvature: float, n_1: float, n_2: float):
        """
        Constructor for SphericalRefraction class.

        Args:
            z_0 (float): position of surface along the optical (z-axis).
            curvature (float): the curvature of the surface - for spherical refractor, curvature = 1/R 
                where R is the radius of curvature.
            n_1 (float): Refractive index of media before surface.
            n_2 (float): Refractive index of media behind surface.
            aperture (float): The circular aperture of the surface.
        """
        self.__z_0 = z_0
        self.__aperture = aperture
        self.__curvature = curvature
        self.__curvature_mag = np.abs(curvature)
        self.__n_1 = n_1
        self.__n_2 = n_2
        
    # Getters for the properties
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
        """
        Finds the point of intercection between the incident ray and the surface

        Args:
            ray (Ray): The ray incident on the surface

        Returns:
            array/None: The point of intersection, or None if the ray does not intersect the surface.
        """
        R = 1. / self.__curvature
        # Vector between the ray position and the optical axis intercept of the surface
        r = ray.pos() - np.array([0., 0., self.__z_0 + R]) 
        b = np.dot(ray.direc(), r)
        
        # Discriminant
        disc = b**2 - (np.dot(r, r) - R**2)
        if disc < 0:
            # No intersection
            return None

        d = [0., 0.]
        sqrt_disc = np.sqrt(disc)
        # Finding both intersections (quadratic formula)
        d[0] = -b + sqrt_disc
        d[1] = -b - sqrt_disc
        
        d.sort()        
        # choosing the right intersection
        if d[0] < 1e-8:
            d[0] = d[1]
            if d[1] < 1e-8:   
                return None
                
        d = d[0]
        # Finds intercept using vector straight line formula
        intercept = ray.pos() + d * ray.direc()
        
        if intercept[0]**2 + intercept[1]**2 > (self.__aperture)**2:
            # If intercept is beyond aperture
            return None
        return intercept
    
    def propagate_ray(self, ray):
        """
        Propagates the ray through the refracting surface making use of the physics.py module.

        Args:
            ray (Ray): The ray to be propagated

        Returns:
            Ray/None: The propagated Ray or None if there is no propagation.
        """
        intercept = self.intercept(ray)
        if intercept is None:
            # no intersection = no propagation
            return None
        
        R = 1. / self.__curvature
        # The normal vector depends on the orientation of the surface
        if self.__curvature < 0:
            normal = - (intercept - np.array([0., 0., self.__z_0 + R]))
        else: 
            normal = intercept - np.array([0., 0., self.__z_0 + R])
        normal /= np.linalg.norm(normal)
    
        refracted_direc = refract(direc = ray.direc(), normal = normal, n_1 = self.__n_1, n_2 = self.__n_2)
        if refracted_direc is None:
            # no propagation - TIR
            return None
        
        ray.append(intercept, refracted_direc)
        return ray  
        
    def focal_point(self) -> float:
        """
        Finds the focal point of the surface along the optical axis

        Returns:
            float: The focal point of the surface.
        """
        R = 1. / self.__curvature_mag
        z = self.__z_0 + (self.__n_2 * R) / (self.__n_2 - self.__n_1) # lensmakers formula
        return z
        
    def plot_surface(self, ax, resolution=100):
        """
        Plots the surface to the input axis.

        Args:
            ax (Axes): The matplotlib axis object to plot to
            resolution (int, optional): Number of points included between the bounds of the surface. Defaults to 100.
        """
        r_max = self.__aperture / 2
        curvature = self.__curvature

        x = np.linspace(-r_max, r_max, resolution)
        y = np.linspace(-r_max, r_max, resolution)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        # Points within aperture
        mask = R2 <= r_max**2

        # At surface position
        Z = np.zeros_like(X) + self.__z_0
        if curvature != 0:
            # Adjusting for curvature sag
            R = 1 / self.__curvature_mag
            Z[mask] += R - np.sqrt(R**2 - R2[mask])
        else:
            Z[mask] += 0  

        Z[~mask] = np.nan

        ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan', rstride=1, cstride=1, linewidth=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
class PlaneRefraction(OpticalElement):
    """
    Creates a planar refracting surface, inheriting from OpticalElement
    """
    def __init__(self, z_0: float, aperture: float, n_1: float, n_2: float):
        """
        Constructor for the surface.

        Args:
            z_0 (float): The intersection of the surface with the optical (z-axis)
            aperture (float): The circular aperture of the surface
            n_1 (float): Refractive index of media before surface.
            n_2 (float): Refractive index of media behind surface.
        """
        self.__z_0 = z_0
        self.__n_1 = n_1
        self.__n_2 = n_2
        self.__aperture = aperture
        
    # Getters for the properties
    def z_0(self):
        return self.__z_0
    
    def n_1(self):
        return self.__n_1
    
    def n_2(self):
        return self.__n_2
    
    def aperture(self):
        return self.__aperture
    
    def intercept(self, ray):
        """
        Finds the intersection of a ray with the suraface.

        Args:
            ray (Ray): The ray intersecting the surface.

        Returns:
            array/None: The intersection point where the ray meets the surface or None if there is no intersection.
        """
        if ray.direc()[2] == 0: 
            # Parallel Ray
            return None
        k = (self.__z_0 - ray.pos()[2]) / ray.direc()[2]
        # finds the intersection using vector straight line formulae
        intercept = ray.pos() + k * ray.direc()
        
        if intercept[0]**2 + intercept[1]**2 > (self.__aperture/2)**2: 
            # Check if intersection is within aperture.
            return None
        
        return intercept
    
    def propagate_ray(self, ray):
        """
        Propagates a ray incident to the surface.

        Args:
            ray (Ray): Ray to be propagated

        Returns:
            Ray/None: The propagated Ray object or None if there is no propagation
        """
        intercept = self.intercept(ray)
        if intercept is None:
            # No propagation
            return None
        normal = np.array([0., 0., 1.])
        if np.dot(ray.direc(), normal) > 0:
            # Depending on orientation of surface compared to travel of ray
            normal = -normal
        refracted_direc = refract(ray.direc(), normal, self.__n_1, self.__n_2)
        if refracted_direc is None:
            # No propagation due to TIR
            return None
        
        ray.append(intercept, refracted_direc)   
        return ray     
        
    def focal_point(self) -> float:
        """
        Returns the focal point of the planar surface which is infinite
        """
        return math.inf
    
    def plot_surface(self, ax, resolution=100):
        """
        Plots the surface on a given matplotlib axis

        Args:
            ax (Axes): Matplotlib axis object to plot onto.
            resolution (int, optional): The number of points used for each direction of the surface. Defaults to 100.
        """
        r_max = self.__aperture / 2
        x = np.linspace(-r_max, r_max, resolution)
        y = np.linspace(-r_max, r_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # at optical axis position
        Z = np.zeros_like(X) + self.__z_0

        ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan', rstride=1, cstride=1, linewidth=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


class OutputPlane(OpticalElement):
    """
    An optical element that represents an invisible barrier where the optical rays terminate.
    """
    def __init__(self, z_0: float):
        """
        Constructor for OutputPlane

        Args:
            z_0 (float): The intersection of the plane with the optical z-axis.
        """
        self.__z_0 = z_0
    
    # Getter for properties
    def z_0(self):
        return self.__z_0
    
    def intercept(self, ray):
        """
        Finds the intersection point of any ray with the surface

        Args:
            ray (Ray): Incident ray

        Returns:
            array/None: The point of intersection between the ray and the surface, or None if there is no intersection
        """
        if ray.direc()[2] == 0: 
            # Parallel ray
            return None
        k = (self.__z_0 - ray.pos()[2]) / ray.direc()[2]
        # Finds intersection point using straight line vector formula
        intercept = ray.pos() + k * ray.direc()
        return intercept
    
    def propagate_ray(self, ray):
        """
        Propagates the ray through the surface.

        Args:
            ray (Ray): Incident Ray to be propagated

        Returns:
            Ray/None: propagated ray or none if the ray does not intersect.
        """
        intercept = self.intercept(ray)
        if intercept == None:
            return None
        ray.append(intercept, ray.direc()) # unchanged direction - not refractor
        return ray 