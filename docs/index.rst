.. Blacklight Fork documentation master file, created by sphinx-quickstart on Tue Jul 18 09:26:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Blacklight Fork's documentation!
===========================================

Hypothesis
==========

The hypothesis of this project is that DNN topologies will converge to either a local maximum or an absolute maximum over the evolution process, offering better performance than a DNN with randomly selected topology. For this experiment, the project will use equivalent activation functions (ReLU) and SGD for back-propagation, holding everything except the topology constant. Updated documentation coming soon.

Methodology
===========

The project utilizes a genetic algorithm to evolve the topology of the DNN. The algorithm starts with a randomly generated population of DNN topologies and evaluates their fitness using the accuracy of the model. The fittest individuals are selected for reproduction, while the weaker ones are discarded. The offspring of the selected individuals are then created through crossover and mutation. This process is repeated for a specified number of generations, and the best-performing topology is chosen as the final output.

Blacklight AutoML
=================
.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   genetic.populations.FeedForward

Blacklight Genetic
==================

Genetic Base Population Class
****************************

.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   genetic.base.Population

Genetic Base Individual Class
****************************

.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   genetic.base.Individual

Genetic Base Chromosome Class
****************************

.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   genetic.base.Chromosome

Populations
==================
.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   genetic.populations.FeedForward


Individuals
***********

.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   genetic.individuals.FeedForwardIndividual

Chromosomes
***********

.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   genetic.chromosomes.FeedForwardChromosome

Blacklight Engine
==================

Model Creator
==================
.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   engine.model_creator

Model Options
==================
.. currentmodule:: blacklight

.. autosummary::
   :nosignatures:
   :template: class.rst

   engine.model_options

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2

   Installation Guide <installation-guide>
