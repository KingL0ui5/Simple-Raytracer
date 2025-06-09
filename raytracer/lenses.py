"""
A module to explore lenses in the raytracer setup 
L Liu 25/05/2025
"""

from .elements import SphericalRefraction, PlaneRefraction, OpticalElement
from .rays import Ray
    
class PlanoConvex(OpticalElement):
    """
    A plano-convex lens - with one flat surface (PlanarRefraction) and one convex surface (SphericalRefraction)
    """
    def __init__(self, z_0: float, curvature1: float, curvature2: float, n_inside: float, n_outside: float, thickness: float, aperture: float):
        """
        Constructor for the PlanoConvex lens.

        Args:
            z_0 (float): position of the lens along the optical (z) axis
            curvature1 (float): curvature of the first surface
            curvature2 (float): curvature of the second surface
            n_inside (float): refractive index of the lens material
            n_outside (float): refractive index of the surrounding medium
            thickness (float): thickness of the lens
            aperture (float): aperture of the lens
        """
        self.__z_0 = z_0
        self.__n_inside = n_inside
        self.__n_outside = n_outside
        self.__thickness = thickness
        self.__aperture = aperture 
        self.__curvature1 = curvature1
        self.__curvature2 = curvature2
        
        front = SphericalRefraction(
            z_0 = z_0,
            n_1 = n_outside,
            n_2 = n_inside,
            curvature = curvature1,
            aperture = aperture
        )
        
        back = SphericalRefraction(
            z_0 = z_0 + thickness,
            curvature = curvature2,
            n_1 = n_inside,
            n_2 = n_outside,
            aperture = aperture
        )
        
        self.__surfaces = [front, back]

    # Getters for the properties of the lens
    def z_0(self):
        return self.__z_0

    def aperture(self):
        return self.__aperture

    def n_inside(self):
        return self.__n_inside
    
    def n_outside(self):
        return self.__n_outside

    def thickness(self):
        return self.__thickness
    
    def curvature1(self):
        return self.__curvature1
    
    def curvature2(self):
        return self.__curvature2

    def propagate_ray(self, ray: Ray):
        """
        Propagate a ray through the lens by passing it through each surface.

        Args:
            ray (Ray): The ray to be propagated

        Returns:
            Ray: The propagated ray, or None if the ray is not propagated.
        """
        for surface in self.__surfaces:
            next_ray = surface.propagate_ray(ray)
            if next_ray is None:
                return None
            ray = next_ray
        return ray
    
    def optical_power(self) -> float:
        """
        Calculate the optical power of the lens using the thick-lens lensmaker's formula.

        Returns:
            Optical Power (float): The optical power of the lens.
        """
        N = self.n_inside() / self.n_outside()
        
        if self.curvature1() == 0 or self.curvature2() == 0:
            return (N - 1) * (self.curvature1() - self.curvature2())

        R1 = 1. / self.curvature1()
        R2 = 1. / self.curvature2()

        # Thick-lens lensmaker's formula
        D = (N - 1) * (self.curvature1() - self.curvature2()) \
            + ((N - 1)**2 * self.thickness()) / (N * R1 * R2)
        return D
    
    def focal_point(self) -> float:
        """
        Calculate the focal point of the lens based on its optical power, thickness and position.

        Returns:
            Focal Point (float): The focal point of the lens.
        """
        front, back = self.__surfaces
        
        if isinstance(front, SphericalRefraction) and isinstance(back, PlaneRefraction):
            shift = - self.thickness() / self.n_inside() 
            return 1. / self.optical_power() + self.z_0() + self.thickness() + shift 
        
        elif isinstance(front, PlaneRefraction) and isinstance(back, SphericalRefraction):
            return 1. / self.optical_power() + self.z_0() + self.thickness() 
        return 0.
    
    def plot_lens(self, ax):
        """
        Plot the lens surfaces on a given axis.

        Args:
            ax (fig ax): The axis on which to plot the lens surfaces.
        """
        self.__surfaces[0].plot_surface(ax = ax)
        self.__surfaces[1].plot_surface(ax = ax)   
 
class BiConvex(OpticalElement):
    """
    A bi-convex lens - with two convex surfaces (SphericalRefraction)
    """
    def __init__(self, z_0: float, curvature1: float, curvature2: float, n_inside: float, n_outside: float, thickness: float, aperture: float):
        """
        Constructor for the BiConvex lens.
        
        Args:
            z_0 (float): position of the lens along the optical (z) axis
            curvature1 (float): curvature of the first surface
            curvature2 (float): curvature of the second surface
            n_inside (float): refractive index of the lens material
            n_outside (float): refractive index of the surrounding medium
            thickness (float): thickness of the lens
            aperture (float): aperture of the lens
        """
        self.__z_0 = z_0
        self.__n_inside = n_inside
        self.__n_outside = n_outside
        self.__thickness = thickness
        self.__aperture = aperture 
        self.__curvature1 = curvature1
        self.__curvature2 = curvature2
        
        front = SphericalRefraction(
            z_0=z_0,
            curvature=curvature1,
            n_1=n_outside,
            n_2=n_inside,
            aperture=aperture
        )
        back = SphericalRefraction(
            z_0=z_0 + thickness,
            curvature=curvature2,
            n_1=n_inside,
            n_2=n_outside,
            aperture=aperture
        )
        
        self.__surfaces = [front, back]

    # Getters for the properties of the lens
    def z_0(self):
        return self.__z_0       
    
    def aperture(self):
        return self.__aperture

    def n_inside(self):
        return self.__n_inside  

    def n_outside(self):
        return self.__n_outside

    def thickness(self):
        return self.__thickness

    def curvature1(self):
        return self.__curvature1
    
    def curvature2(self):
        return self.__curvature2
    
    def propagate_ray(self, ray: Ray):
        """
        Propagate a ray through the lens by passing it through each surface.

        Args:
            ray (Ray): The ray to be propagated

        Returns:
            Ray: The propagated ray, or None if the ray is not propagated.
        """
        for surface in self.__surfaces:
            next_ray = surface.propagate_ray(ray)
            if next_ray is None:
                return None
            ray = next_ray
        return ray
    
    
    # def optical_power(self) -> float:
    #     D = (self.n_inside() / self.n_outside()) * (self.curvature1() - self.curvature2())
    #     return D 
    
    # def focal_point(self) -> float: 
    #     return 1. / self.optical_power() + self.z_0() + self.thickness()