"""
A module to explore lenses in the raytracer setup 
L Liu 25/05/2025
"""

from abc import ABC, abstractmethod
import numpy as np

from raytracer.elements import SphericalRefraction, PlaneRefraction, OpticalElement
from raytracer.rays import Ray

class LensMaker:
    def __new__(
        cls,
        z_0: float,
        curvature1: float,
        curvature2: float,
        n_inside: float,
        n_outside: float,
        thickness: float,
        aperture: float
    ):
        """
        Factory dispatch: instantiate the correct subclass with appropriate parameters.
        """
        if curvature1 == 0 and curvature2 != 0:
            return PlanoConvex(z_0=z_0, curvature=curvature2, n_inside=n_inside, n_outside=n_outside, thickness=thickness, aperture=aperture)
        elif curvature2 == 0 and curvature1 != 0:
            return ConvexPlano(z_0, curvature1, n_inside, n_outside, thickness, aperture)
        else:
            return BiConvex(z_0, curvature1, curvature2, n_inside, n_outside, thickness, aperture)
        

class Lens(OpticalElement):
    """
    A base class for lenses, which are optical elements that refract light.
    This class is not intended to be instantiated directly.
    """
    
    def __init__(
        self,
        z_0: float,
        n_inside: float,
        n_outside: float,
        thickness: float,
        aperture: float
    ):
        """
        Initialize common lens parameters.

        Args:
            z_0 (float): position of the lens along the optical (z) axis
            curvature1 (float): curvature of the first surface
            curvature2 (float): curvature of the second surface
            n_inside (float): refractive index of the lens material
            n_outside (float): refractive index of the surrounding medium
            thickness (float): thickness of the lens
            aperture (float): aperture of the lens
        """
        self._z_0 = z_0
        self._n_inside = n_inside
        self._n_outside = n_outside
        self._thickness = thickness
        self._aperture = aperture
        self._surfaces = []
    
    def focal_point(self):
        """
        Calculate the focal point of the lens based on its optical power, thickness and position.

        Returns:
            Focal Point (float): The focal point of the lens.
        """
        raise NotImplementedError("Focal point must be implemented in derived classes")
    
    def plot_lens(self, ax):
        """
        Plot the lens surfaces on a given axis.

        Args:
            ax (fig ax): The axis on which to plot the lens surfaces.
        """
        self._surfaces[0].plot_surface(ax=ax)
        self._surfaces[1].plot_surface(ax=ax)
        
        
    # Getter methods for base attributes
    def z_0(self) -> float:
        return self._z_0

    def n_inside(self) -> float:
        return self._n_inside

    def n_outside(self) -> float:
        return self._n_outside

    def thickness(self) -> float:
        return self._thickness

    def aperture(self) -> float:
        return self._aperture
    
    def propagate_ray(self, ray: Ray):
        """
        Propagate a ray through the lens by passing it through each surface.

        Args:
            ray (Ray): The ray to be propagated

        Returns:
            Ray: The propagated ray, or None if the ray is not propagated.
        """
        for surface in self._surfaces:
            next_ray = surface.propagate_ray(ray)
            if next_ray is None:
                return None
            ray = next_ray
        return ray


class PlanoConvex(Lens):
    """
    A plano-convex lens - with one flat surface (PlanarRefraction) and one convex surface (SphericalRefraction)
    """
    def __init__(
        self,
        z_0: float,
        curvature: float,
        n_inside: float,
        n_outside: float,
        thickness: float,
        aperture: float,
    ):
        """
        Constructor for the PlanoConvex lens.

        Args:
            z_0 (float): position of the lens along the optical (z) axis
            curvature (float): curvature of the convex surface
            n_inside (float): refractive index of the lens material
            n_outside (float): refractive index of the surrounding medium
            thickness (float): thickness of the lens
            aperture (float): aperture of the lens
        """
        super().__init__(z_0=z_0, n_inside=n_inside, n_outside=n_outside, thickness=thickness, aperture=aperture)
        self.__curvature = curvature
        self.__radius = 1. / curvature

        front = PlaneRefraction(
            z_0=z_0,
            n_1=n_outside,
            n_2=n_inside,
            aperture=aperture
        )
        back = SphericalRefraction(
            z_0=z_0 + thickness,
            curvature=curvature,
            n_1=n_inside,
            n_2=n_outside,
            aperture=aperture
        )
        self._surfaces = [front, back]
        
    def focal_length(self):
        """
        Calculate the focal length of the lens.

        Returns:
            float: the focal length of the lens.
        """
        return 1 / (self._n_inside / self._n_outside - 1) * np.abs(self.__radius)
    
    def focal_point(self) -> float:
        """
        Calculate the focal point of the lens based on its optical power, thickness and position.

        Returns:
            Float: The focal point of the lens.
        """
        return self._z_0 + self.focal_length() + self._thickness
    
    def curvature(self):
        return self.__curvature

class ConvexPlano(Lens):
    """
    A convex-plano lens - with one convex surface (SphericalRefraction) and one flat surface (PlaneRefraction)
    """
    def __init__(
        self,
        z_0: float,
        curvature: float,
        n_inside: float,
        n_outside: float,
        thickness: float,
        aperture: float,
    ):
        """
        Constructor for the ConvexPlano lens.

        Args:
            z_0 (float): position of the lens along the optical (z) axis
            curvature (float): curvature of the convex surface
            n_inside (float): refractive index of the lens material
            n_outside (float): refractive index of the surrounding medium
            thickness (float): thickness of the lens
            aperture (float): aperture of the lens
        """
        super().__init__(z_0=z_0, n_inside=n_inside, n_outside=n_outside, thickness=thickness, aperture=aperture)
        self.__curvature = curvature
        self.__radius = 1. / curvature

        front = SphericalRefraction(
            z_0=z_0,
            curvature=curvature,
            n_1=n_outside,
            n_2=n_inside,
            aperture=aperture
        )
        back = PlaneRefraction(
            z_0=z_0 + thickness,
            n_1=n_inside,
            n_2=n_outside,
            aperture=aperture
        )
        self._surfaces = [front, back]
        
    def focal_length(self):
        """
        Calculate the focal length of the lens.

        Returns:
            float: the focal length of the lens.
        """
        return 1 / (self._n_inside / self._n_outside - 1) * np.abs(self.__radius)
    
    def focal_point(self) -> float:
        """
        Calculate the focal point of the lens based on its focal length, thickness and position.

        Returns:
            Float: The focal point of the lens.
        """
        h = self._thickness / self._n_inside
        return self._z_0 + self.focal_length() + self._thickness - h
    
    # Getter for curvature attribute
    def curvature(self):
        return self.__curvature
    
class BiConvex(Lens):
    """
    A bi-convex lens - with two convex surfaces (SphericalRefraction)
    """
    def __init__(
        self,
        z_0: float,
        curvature1: float,
        curvature2: float,
        n_inside: float,
        n_outside: float,
        thickness: float,
        aperture: float,
    ):
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
        super().__init__(z_0=z_0, n_inside=n_inside, n_outside=n_outside, thickness=thickness, aperture=aperture)
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
        self._surfaces = [front, back]   
        
    def focal_length(self):
        N = self._n_inside / self._n_outside
        D = (N - 1) * (self.__curvature1 - self.__curvature2)
        return 1. / D

    def focal_point(self) -> float:
        """
        Calculate the focal point of the lens based on its optical power, thickness and position.

        Returns:
            Focal Point (float): The focal point of the lens.
        """
        return self._z_0 + self.focal_length() + self._thickness
    
    # Getters for curvature attributes
    def curvature1(self):
        return self.__curvature1
    
    def curvature2(self):
        return self.__curvature2