"""
A module to make further investigations using the raytracer code
"""
from raytracer.lenses import LensMaker
from raytracer.rays import RayBundle, DivergingRayBundle
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np
    
def spherical_abberation():
    """
    This function is used to investigate spherical abberation by iterating through different directions of off-axis rays, 
    finding and comparing their focal points with each other and the focal point of the lens.
    """
    
    z_0 = 50.0
    lens = LensMaker(
        z_0=z_0,
        curvature1=0.02,
        curvature2=0.02,
        n_inside=1.5,
        n_outside=1.0,
        thickness=5.0,
        aperture=10.0
    )
    calc_focal_point = lens.focal_point()

    bundle = DivergingRayBundle(
        spread=1.0,
        nrings=2,
        multi=6,
        maxangle=np.pi/4,
        nangle=10
    )
    sim_results = [focus(rays, zbounds=[0, z_0]) for rays in bundle.rays()]
    sim_focal_points, sim_RMS_vals = zip(*sim_results)

    # angles in rads for each bundle set
    angles = bundle.angles()

    plt.figure()
    plt.scatter(angles, sim_focal_points, marker='o', label='Simulated focal lengths')
    plt.scatter(angles, sim_RMS_vals, marker='x', label='RMS spot size')
    plt.xlabel('Launch angle (rad)')
    plt.ylabel('Distance or RMS (mm)')
    plt.title('Focal Point and RMS vs Launch Angle')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def focus(bundle: RayBundle, zbounds: list[float]) -> tuple:
    """
    Finds the focus of the rays by minmising the RMS spread of rays at a z-coordinate

    Args:
        bundle (RayBundle): the ray bundle in consideration.
        zbounds (list[float]): the z-coordinate window to consider.

    Returns:
        tuple[float]: the z position and minimum RMS.
    """
    res = minimize_scalar(lambda z: rms_z(bundle, z), bounds=zbounds, method='bounded')
    return res.x, res.fun 
    

def rms_z(bundle: RayBundle, z: float) -> float:
    """
    Find the rms spread of points value at position z along coordinate axis

    Args:
        bundle (RayBundle): bundle of rays
        z (float): z-coordinate or distance along optical axis

    Returns:
        float: rms spread of rays at coordinate z.
    """
    for ray in bundle.rays():
        pos = ray.pos()
        direc = ray.direc()

        if direc[2] == 0:
            # ray is parallel to z axis
            continue  

        t = (z - pos[2]) / direc[2]
        new_pos = pos + t * direc

        # temporarily add vertex
        ray.append(new_pos, direc)

    return bundle.rms()

if __name__ == "__main__":
    # Test spherical_abberation
    spherical_abberation()
    plt.show()
    