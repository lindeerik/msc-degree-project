"""
Setup script for the msc_degree_project package.
"""

from setuptools import setup, find_packages

setup(
    name="msc_degree_project",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
