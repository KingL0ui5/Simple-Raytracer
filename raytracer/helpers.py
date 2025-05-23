"""
Helper functions for the raytracer.
L Liu 23/05
"""
import numpy as np

class Utils:
    @staticmethod
    def validate_vector(vec, name: str) -> None:
        if len(vec) != (3):
            raise ValueError(f"{name!r} must be length 3, got length {len(vec)}")

    @staticmethod
    def format_vector(vec) -> np.ndarray:
        return np.array(vec, dtype=float)
    
    @staticmethod
    def normalise_vector(vec):
        return vec / np.linalg.norm(vec)

    @classmethod
    def validate_and_format(cls, vec, name: str) -> np.ndarray:
        cls.validate_vector(vec, name)
        return cls.format_vector(vec)
