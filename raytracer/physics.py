"""
A module for refractive physics calculations using snell's law.
L Liu 23/05
"""
import numpy as np
from raytracer.helpers import Utils as hlp 
        
def refract(direc: np.ndarray, normal: np.ndarray, n_1: float, n_2: float):
    """
    Uses Snells Law to refract a light ray through a surface given the direction of the ray and the normal vector of the surface.

    Args:
        direc (np.ndarray): direction of incident ray
        normal (np.ndarray): direction of normal vector
        n_1 (float): refractive index of initial medium
        n_2 (float): refractive index of new medium

    Returns:
        array: the refracted direction of the ray.
    """
    # Validations
    hlp.validate_vector(direc, "Direc")
    hlp.validate_vector(normal, "Normal")
    direc = hlp.normalise_vector(direc)
    normal = hlp.normalise_vector(normal)

    eta = n_1 / n_2
    cos_theta_1 = -np.dot(direc, normal)
    sin2_theta_2 = eta**2 * (1 - cos_theta_1**2)

    if sin2_theta_2 > 1.:
        # Total Interal Reflection
        return None
    
    cos_theta_2 = np.sqrt(1 - sin2_theta_2)
    # Vector form of refraction formula
    ref_direc = eta * direc + (eta * cos_theta_1 - cos_theta_2) * normal 
    hlp.normalise_vector(ref_direc)
    
    return ref_direc

def reflect(direc: np.ndarray, normal: np.ndarray):
    """
    Reflects a light ray off a surface given the direction of the ray and the normal vector of the surface.

    Args:
        direc (np.ndarray): direction of incident ray
        normal (np.ndarray): direction of normal vector

    Returns:
        array: the reflected direction of the ray.
    """
    # Validations
    hlp.validate_vector(direc, "Direc")
    hlp.validate_vector(normal, "Normal")
    direc = hlp.normalise_vector(direc)
    normal = hlp.normalise_vector(normal)

    # Vector form of reflection formula
    ref_direc = direc - 2 * np.dot(direc, normal) * normal
    ref_direc = hlp.normalise_vector(ref_direc)
    return ref_direc
