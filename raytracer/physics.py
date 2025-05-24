"""
A module for refractive physics calculations using snell's law.
L Liu 23/05
"""
import numpy as np
from raytracer.helpers import Utils as hlp 
        
def refract(direc, normal, n_1, n_2): # test
    hlp.validate_vector(direc, "Direc")
    hlp.validate_vector(normal, "Normal")
    direc = hlp.normalise_vector(direc)
    normal = hlp.normalise_vector(normal)

    eta = n_1 / n_2
    cos_theta_1 = -np.dot(direc, normal)
    sin2_theta_2 = eta**2 * (1 - cos_theta_1**2)

    if sin2_theta_2 > 1.:
        return None

    cos_theta_2 = np.sqrt(1 - sin2_theta_2)
    ref_direc = eta * direc + (eta * cos_theta_1 - cos_theta_2) * normal 
    
    # ref_direc = (np.cos(theta_2 - theta_1) * normal) + (np.cos(theta_2) * direc)

    hlp.normalise_vector(ref_direc)
    
    return ref_direc
