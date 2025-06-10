"""
A module to make further investigations using the raytracer code
"""
from raytracer.elements import SphericalReflection, OutputPlane
from raytracer.rays import RayBundle, DivergingRayBundle
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import numpy as np
    
def spherical_abberation():
    """
    This function is used to investigate spherical abberation by iterating through different directions of off-axis rays, 
    finding and comparing their focal points with each other and the focal point of the lens.
    """
    
    z_0_mirror = 50.
    mirror = SphericalReflection(z_0=z_0_mirror, curvature=-0.02, aperture=100.)
    calc_focal_point = mirror._z_0 - mirror.focal_length()
    
    z_0_out = calc_focal_point - 50.
    output = OutputPlane(z_0=z_0_out)
    

    div_bundle = DivergingRayBundle(
        spread=1.0,
        nrings=1,
        multi=2,
        maxangle=np.pi/10,
        nangle=100
    )
    div_bundle.propagate_bundle([mirror, output])
    
    sim_results = [focus(bundles, zbounds=[z_0_out, z_0_mirror]) for bundles in div_bundle.bundles()]
    sim_focal_points, sim_RMS_vals = zip(*sim_results)

    angles = div_bundle.angles()

    angles = np.array(angles) # shape (n_angles,)
    focals = np.array(sim_focal_points) # shape (n_angles * multi,)
    rms = np.array(sim_RMS_vals) # same shape

    n_angles = len(angles)
    multi = len(focals) // n_angles # number of bundles per angle (== rays per ring)

    focals_by_angle = focals.reshape(n_angles, multi)
    rms_by_angle = rms.reshape(n_angles, multi)

    mean_focals = focals_by_angle.mean(axis=1)
    mean_rms = rms_by_angle.mean(axis=1)

    # Launch angle vs distance from mirror
    plt.figure(figsize=(6,4))
    plt.scatter(angles[1:], mean_focals[1:], marker='o', label='Mean focal length')
    plt.scatter(0, calc_focal_point, marker = 'x', color = 'red')
    plt.xlabel('Launch angle (rad)')
    plt.ylabel('Distance (mm)')
    plt.title('Spherical Aberration vs Launch Angle')
    plt.legend()
    plt.grid(True)
    
    # RMS values 
    plt.figure()
    plt.scatter(angles, mean_rms, marker='x', label='Mean RMS spot size')
    plt.xlabel('Launch angle (rad)')
    plt.ylabel('Min RMS (mm)')
    plt.title('Min RMS vs Launch Angle')
    plt.legend()
    plt.grid(True)
    

    
    # Ray tracing figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore
    mirror.plot_surface(ax)
    ax.scatter([0], [0], [calc_focal_point], c='red', marker='x')
    ax = div_bundle.track_plot(ax)
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
    
    
    # To do: fix focal length for biconvex, fix data representation 