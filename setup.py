from setuptools import setup, find_packages

setup(
    name='mlbase',
    author='Jason Stock',
    packages=find_packages(exclude=('test',)),
)
