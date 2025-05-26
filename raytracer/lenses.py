"""
A module to explore lenses in the raytracer setup 
L Liu 25/05/2025
"""

from .elements import SphericalRefraction, PlaneRefraction, OpticalElement
from .rays import Ray

class Lens(OpticalElement):
    def __init__(self, z_0: float, surfaces: list[OpticalElement], n_inside: float, n_outside: float, thickness: float, aperture: float):
        self._z0 = z_0
        self._surfaces = surfaces
        self._n_in = n_inside
        self._n_out = n_outside
        self._thick = thickness
        self._ap = aperture

    def z_0(self):
        return self._z0

    def aperture(self):
        return self._ap

    def n_inside(self):
        return self._n_in

    def n_outside(self):
        return self._n_out

    def thickness(self):
        return self._thick

    def propagate_ray(self, ray: Ray) -> Ray | None:
        for surf in self._surfaces:
            ray = surf.propagate_ray(ray)
            if ray is None:
                return None
        return ray

    def focal_point(self) -> float:
        return self._surfaces[-1].focal_point()


class PlanoConvex(Lens):
    def __init__(self, z_0: float, curvature: float, n_inside: float, n_outside: float, thickness: float, aperture: float):
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
        super().__init__(
            z_0=z_0,
            surfaces=[front, back],
            n_inside=n_inside,
            n_outside=n_outside,
            thickness=thickness,
            aperture=aperture
        )


class BiConvex(Lens):
    def __init__(self, z_0: float, curvature1: float, curvature2: float, n_inside: float, n_outside: float, thickness: float, aperture: float):
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
        super().__init__(
            z_0=z_0,
            surfaces=[front, back],
            n_inside=n_inside,
            n_outside=n_outside,
            thickness=thickness,
            aperture=aperture
        )
