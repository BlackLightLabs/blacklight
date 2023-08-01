Genetic Algorithms in AutoML
============================

This project aims to use Genetic Algorithms to optimize the topologies of Deep Neural Networks (DNNs) and explore new possibilities that traditional optimization techniques might overlook. The fitness function of the algorithm is the accuracy of the model, and the genes represent the individual topologies.

Installation
------------

Make sure you have Python 3.9 or higher installed (not greater than 3.11).

Windows, Linux
~~~~~~~~~~~~~~

Create a new virtual environment::

    pip install -m virtualenv
    python -m venv your_virtual_env_name
    your_virtual_env_name\Scripts\activate

Install Tensorflow::

    pip install tensorflow

Install the package::

    pip install blacklight

MacOS (Intel)
~~~~~~~~~~~~~

Create a new virtual environment::

    pip install -m virtualenv
    python -m venv your_virtual_env_name
    your_virtual_env_name\Scripts\activate

Install Tensorflow::

    pip install tensorflow-macos
    pip install tensorflow-metal

Install the package::

    pip install blacklight

MacOS (Apple Silicon)
~~~~~~~~~~~~~~~~~~~~~

Download Miniconda from: https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

Install Miniconda::

    Navigate to downloads folder: cd ~/Downloads
    bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda

Activate Miniconda::

    source ~/miniconda/bin/activate

Install TensorFlow dependencies::

    conda install -c apple tensorflow-deps

Install TensorFlow::

    pip install tensorflow-macos
    pip install tensorflow-metal

Install the package::

    pip install blacklight
