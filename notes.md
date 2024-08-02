# :warning: These are maintainer notes that will be deleted :warning:

# Plan

## ./src/blacklight/engine/model_creator.py

This file will be responsible for creating a model based on the config provided to it.
During later iterations of the the program (that is iterations as in run cycles) the create model function will create models based on the provided genes.
It is important the the function creates the models to the specifications of the genes while not bypassing the restrictions set in the model config.

## ./src/blacklight/engine/tests

Before moving on to further functionality test [./src/blacklight/engine/model_creator.py](model_creator)

## ./src/blacklight/genetic/

Create chromosomes, individuals and populations and their respective functionalities. It might be advantagious for code quality to merge chromosomes and individuals.

# Plans for further development
- Improve the methods for topology optimization. This could entail multiple complete rewrites of the section of the library.
   - Consider different options for how the genes representing a model can be, modeled. 
      - If a bring your own model solution can be devised in which the user provides their own model to the `create_model` function inside `model_creator.py`, that would be ideal. 
         in this solution the library user would define their own model using whatever means Pytorch allows, this model would then be internally (or not) serialized into a form
         that the library can then edit and iterate over.
- autoML
