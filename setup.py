from setuptools import find_packages, setup

setup(
    name='blacklight',
    packages=find_packages(include=['blacklight']),
    version='0.1.0',
    description='Genetic Deep Neural Network Topology Search',
    author='Cole Agard',
    license='GNU Public',
    test_suite='test',
)
