from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("simple_linear_regression", ["bindings.cpp"])
]

setup(
    name="simple_linear_regression",
    version="0.1.0",
    author="Your Name",
    description="Python bindings for Simple Linear Regression",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
