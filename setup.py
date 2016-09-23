
"""

Setup file for compiling cython extensions

To build cython libraries in place run:
python setup.py build_ext --inplace

# Build wheel and install wheel
python setup.py bdist_wheel
# Install wheel from local file
pip install perfusionslic-0.12-cp27-none-linux_x86_64.whl

"""

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy

Description = """/
Perfusion SLIC
"""

extensions = [
    Extension("maskslic._slic",
        sources=["maskslic/_slic.pyx"],
        include_dirs=[numpy.get_include()]),
    Extension("maskslic.processing",
              sources=["maskslic/processing.pyx",
                       "src/processing.cpp"],
              include_dirs=["src",
                            numpy.get_include()],
              language="c++",
              extra_compile_args=["-std=c++11"])
]

setup(name="maskslic",
      packages=["maskslic"],
      version="0.21",
      description="",
      author='Benjamin Irving',
      author_email='mail@birving.com',
      url='www.birving.com',
      long_description=Description,
      install_requires=['nibabel', 'scikit-image', 'scikit-learn', 'Cython', 'matplotlib'],
      ext_modules=cythonize(extensions)
)
