"""
A module to store the Ray class to describe the rays in the system. 
Created L Liu 19/05
"""
from raytracer.helpers import Utils as hlp
from raytracer.genpolar import rtrings

import numpy as np
import matplotlib.pyplot as plt

class Ray:
    """
    An optical ray.
    """
    def __init__(self, pos: list[float] = [0,0,0], direc: list[int] = [0,0,1]):
        """
        Constructor for the Ray class.

        Args:
            pos (list[float], optional): Initial position of the ray as a 3d vector. Defaults to [0,0,0].
            direc (list[int], optional): Initial direction of the ray as a 3d vector. Defaults to [0,0,1].
        """
        self.__pos = hlp.validate_and_format(pos, "Pos") # Using helper function to check and format the input
        direc = hlp.normalise_vector(direc)
        self.__direc = hlp.validate_and_format(direc, "Dir")

        self.__vertices = [self.__pos] 
    
    # getters for the ray properties
    def pos(self) -> np.ndarray:
        return self.__pos
    
    def direc(self) -> np.ndarray:
        return self.__direc
    
    def append(self, pos, direc) -> None:
        """
        Append a new position and direction to the ray.

        Args:
            pos (3d array): position of the ray as a 3d vector
            direc (3d array): direction of the ray as a 3d vector
        """
        pos = hlp.validate_and_format(pos, "Pos")
        direc = hlp.validate_and_format(direc, "Dir")
        direc = hlp.normalise_vector(direc)
    
        self.__vertices.append(pos) 
        self.__direc = direc
        self.__pos = pos # Update position and direction of the ray
    
    # getter for the verticies (all points) of the ray
    def vertices(self) -> list[np.ndarray]:
        return self.__vertices
    
class RayBundle:
    """
    A bundle of rays arranged in concentric rings using the genpolar module.
    """
    def __init__(self, rmax: float = 5., nrings: int = 5, multi: int = 6):
        """
        Constructor for the RayBundle class.

        Args:
            rmax (float, optional): maximum radius of the bundle. Defaults to 5..
            nrings (int, optional): number of concentric rings in the bundle. Defaults to 5.
            multi (int, optional): number of points in each ring. Defaults to 6.
        """
        if not isinstance(multi, int):
            raise TypeError(f'multi must be of type int, got type {type(multi)}')
        if not isinstance(nrings, int):
            raise TypeError(f'nrings must be of type int, got type {type(nrings)}') 
        
        self.__rays = []
        
        # Generate concentric rings of rays
        for r, theta in rtrings(rmax, nrings, multi): 
            # Convert to cartesian
            x = r * np.cos(theta) 
            y = r * np.sin(theta)
            
            pos = [x, y, 0.]
            direc = [0, 0, 1]

            ray = Ray(pos=pos, direc=direc)
            self.__rays.append(ray)
        
    def propagate_bundle(self, elements):
        """
        Propagate the bundle of rays through a list of optical elements.

        Args:
            elements (OpticalElement): Elements through which the rays are propagated.
        """
        for element in elements:
            for ray in self.__rays:
                element.propagate_ray(ray)
        
    def track_plot(self):
        """
        Create a 3D plot of the vertices of all rays in the bundle.
        
        Returns:
            fig: A matplotlib figure containing the 3D plot of the ray vertices.
        """
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
        ax.set_zlabel('Z')  # type: ignore
        ax.set_title('Ray Vertices Path')
    
        return fig
        
    def rms(self):
        """
        Calculate the RMS (Root Mean Square) of the positions of the rays in the bundle projected onto the XY plane.

        Returns:
            float: The RMS value.
        """
        positions = np.array([ray.pos() for ray in self.__rays])
        squared_dists = np.sum(positions[:, :2] ** 2, axis=1)
        rms = np.sqrt(np.mean(squared_dists))
        return rms


    def spot_plot(self, fig=None):
        """
        Create a 2D spot plot of the positions of the rays in the bundle projected onto the XY plane.

        Returns:
            fig: The figure containing the spot plot.
        """
        x = [ray.pos()[0] for ray in self.__rays]
        y = [ray.pos()[1] for ray in self.__rays]

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(x, y, s=10, color='blue')
            ax.set_xlabel('x position (mm)')
            ax.set_ylabel('y position (mm)')
            ax.set_title('Spot Plot')
            ax.grid(True)
        else:
            fig = fig.scatter(x, y, s=10, color='blue')
        return fig
            
        
            
    
    
        
            
    

        
        