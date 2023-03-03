.. Blacklight documentation master file, created by
   sphinx-quickstart on Wed Mar  1 14:02:52 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Blacklight's API Documentation!
==========================================

.. py:module:: `blacklight.autoML`

.. automodule:: blacklight.autoML
   :no-members:
   :no-inherited-members:

AutoML classes
##############

.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   autoML.FeedForward

AutoML utilities
################

AutoML Populations Classes
**************************

.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   autoML.populations.Population

AutoML Individuals Classes
**************************

.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   autoML.individuals.Individual
   autoML.individuals.FeedForwardIndividual

AutoML Individuals Utility Functions
************************************

.. currentmodule:: blacklight

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: function.rst

    autoML.individuals.get_min_length_chromosome
    autoML.individuals.get_crossover_points_from_num_parents
    autoML.individuals.merge_genes
    autoML.individuals.mutate_dominant_gene


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
