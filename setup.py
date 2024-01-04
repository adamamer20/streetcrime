from setuptools import find_packages, setup

setup(
    name="streetcrime",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version="1.0",
    description="An extension to the Mesa agent-based modeling framework using Pandas DataFrames for enhanced performance, applied to the study of Street Theft and Robbery",
    author="Adam Amer",
    license="MIT License",
)
