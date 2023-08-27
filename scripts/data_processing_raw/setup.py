from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="data_processing_raw", 
                sources=[r"E:\Users\adiad\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\scripts\data_processing_raw\data_processing_raw.pyx"])

setup(ext_modules=cythonize(ext))