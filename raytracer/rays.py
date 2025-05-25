"""
A module to store the Ray class to describe the rays in the system. 
Created L Liu 19/05
"""
from .helpers import Utils as hlp
from .genpolar import rtrings

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Ray:
    def __init__(self, pos: list[float] = [0,0,0], direc: list[int] = [0,0,1]):
        self.__pos = hlp.validate_and_format(pos, "Pos")
        direc = hlp.normalise_vector(direc)
        self.__direc = hlp.validate_and_format(direc, "Dir")

        self.__vertices = [self.__pos]
        
    def pos(self) -> np.ndarray:
        return self.__pos
    
    def direc(self) -> np.ndarray:
        return self.__direc
    
    def append(self, pos, direc) -> None:
        pos = hlp.validate_and_format(pos, "Pos")
        direc = hlp.validate_and_format(direc, "Dir")
        direc = hlp.normalise_vector(direc)
    
        self.__vertices.append(pos)
        self.__direc = direc
        self.__pos = pos
    
    def vertices(self) -> list[np.ndarray]:
        return self.__vertices
    
class RayBundle:
    def __init__(self, rmax: float = 5., nrings: int = 5, multi: int = 6):
        if not isinstance(multi, int):
            raise TypeError(f'multi must be of type int, got type {type(multi)}')
        if not isinstance(nrings, int):
            raise TypeError(f'nrings must be of type int, got type {type(nrings)}')
        
        self.__rays = []
        
        for r, theta in rtrings(rmax, nrings, multi):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            pos = [x, y, 0.]
            direc = [0, 0, 1]

            ray = Ray(pos=pos, direc=direc)
            self.__rays.append(ray)
        
    def propagate_bundle(self, elements):
        for element in elements:
            for ray in self.__rays:
                element.propagate_ray(ray)
        
    def track_plot(self):
        vertices = [v for ray in self.__rays for v in ray.vertices()]
        points = np.vstack(vertices)
        
        x = points[:,0]
        y = points[:, 1]
        z = points[:, 2]    
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, marker='o') 

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') 
        ax.set_title('Ray Vertices Path')
    
        return fig
        

    
        
            
    

        
        