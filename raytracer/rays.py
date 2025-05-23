"""
A module to store the Ray class to describe the rays in the system. 
Created L Liu 19/05
"""
from raytracer.helpers import Utils as hlp
import numpy as np

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
    
    

        
        