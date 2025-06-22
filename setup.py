from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("compute.pyx"),
    include_dirs=[np.get_include()]
)


#from qlpso import compute_distance, softmax
