from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("pine_fast", ["pine_fast.pyx"]),
    Extension("mql5_fast", ["mql5_fast.pyx"]),
]

setup(
    name="backtester_engines",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        },
    ),
)
