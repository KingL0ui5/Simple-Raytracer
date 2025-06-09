"""
Helper functions for validations and other operations for the raytracer.
L Liu 23/05
"""
import numpy as np

class Utils:
    """
    A class of static utility functions useful for vector input validations, formatting and exceptions.
    """
    @staticmethod
    def validate_vector(vec, name: str) -> None:
        """ 
        Validates the length of the vector input

        Args:
            vec (array): Any 3d vector (pos or direc)
            name (str): The name of the vector

        Raises:
            ValueError: If the length of the input is not 3
        """
        if len(vec) != (3):
            raise ValueError(f"{name!r} must be length 3, got length {len(vec)}")

    @staticmethod
    def format_vector(vec) -> np.ndarray:
        # Formats standard array input to np.ndarray
        return np.array(vec, dtype=float)
    
    @staticmethod
    def normalise_vector(vec):
        # normalises the vector input
        return vec / np.linalg.norm(vec)

    @classmethod
    def validate_and_format(cls, vec, name: str) -> np.ndarray:
        # Validates and formats vector input using the two above functions.
        cls.validate_vector(vec, name)
        return cls.format_vector(vec)
