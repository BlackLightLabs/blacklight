# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
for x in os.walk('../../blacklight'):
    sys.path.append(x[0])
sys.path.append(".")
sys.path.append('../')

# autodoc_mock_imports = ["numpy", "pandas", "tensorflow", "sklearn"]
autodoc_default_options = {"members": True, "private-members": True}
add_module_names = False

project = 'Blacklight'
copyright = '2023, Cole Agard'
author = 'Cole Agard'
release = '0.1.6'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



# html_additional_pages = {
#   'index': 'custom_index.html',
#   'installation_guide': 'custom_installation-guide.html',
# }

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'furo.sphinxext',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Furo sidebar customization
html_theme = 'furo'
html_theme_options = {
    'navigation_with_keys': True,
    'sidebar_hide_name': False,  # Display the TOC in the sidebar
    'sidebar_hide_navigation': False,  # Display the navigation links in the sidebar
}

# Add your table of contents here if required
html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/navigation.html',
        'module.html',
    ]
}


# ... (other configurations)
