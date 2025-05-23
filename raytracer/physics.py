"""
A module for refractive physics calculations using snell's law.
L Liu 23/05
"""
import numpy as np
from raytracer.helpers import Utils as hlp 
        
def refract(direc, normal, n_1, n_2):
    hlp.validate_vector(direc, "Direc")
    hlp.validate_vector(normal, "Normal")
    direc = hlp.normalise_vector(direc)
    normal = hlp.normalise_vector(normal)
    
    theta_1 = np.arccos(np.dot(direc, normal))
    sin_theta_2 = (n_1 / n_2) * np.sin(theta_1)
    if sin_theta_2 > 1:
        return None
    
    theta_2 = np.arcsin(sin_theta_2)
    ref_direc = (np.cos(theta_2 - theta_1) * normal) + (np.cos(theta_2) * direc)
    
    ref_direc /= np.linalg.norm(ref_direc)
    
    return (ref_direc)

direc = np.array([0., 0., 1.])
norm_lower = np.array([0., -1., -1.])
norm_lower /= np.linalg.norm(norm_lower)

output = refract(direc=direc, normal=norm_lower, n_1=1.0, n_2=1.5) #, np.array([0., 0.29027623, 0.9569429]))
print(output) 
