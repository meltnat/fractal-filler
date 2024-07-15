"""This file is used to import all the functions from the fractal module."""

from fractal.box_count import dq_from_tensor, fractal_dimension
from fractal.box_count_old import BoxCountOld

__all__ = ["BoxCountOld", "fractal_dimension", "dq_from_tensor"]
