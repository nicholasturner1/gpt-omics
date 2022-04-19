import sys
import setuptools
from setuptools import setup, find_packages


__version__ = '0.0.0'


setup(
    name='gptomics',
    version=__version__,
    description='Making happy (well-documented) CloudVolumes',
    author='Nicholas Turner',
    author_email='nicholaslarryturner@gmail.com',
    url='https://github.com/nicholasturner1/gpt-omics',
    packages=setuptools.find_packages(),
)
