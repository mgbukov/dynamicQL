from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'QL',
  ext_modules = cythonize("Q_learning_c.pyx"),
)