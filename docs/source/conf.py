# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Blacklight'
copyright = '2023, Cole Agard'
author = 'Cole Agard'
release = '0.1.6'

autodoc_mock_imports = ["numpy", "pandas", "tensorflow", "sklearn"]
autodoc_default_options = {"members": True,  "private-members": True}
add_module_names = False
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys
import os
for x in os.walk('../../blacklight'):
  sys.path.append(x[0])
sys.path.append(".")
sys.path.append('../')


extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
