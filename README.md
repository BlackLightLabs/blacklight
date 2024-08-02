# Blacklight  

[![test](https://github.com/BlackLightLabs/blacklight/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/BlackLightLabs/blacklight/actions/workflows/test.yml) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/449f7ff90fcb4340a4c90884d15f700a)](https://www.codacy.com/gh/BlackLightLabs/blacklight/dashboard?utm_source=github.com&utm_medium=referral&utm_content=BlackLightLabs/blacklight&utm_campaign=Badge_Coverage) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/449f7ff90fcb4340a4c90884d15f700a)](https://www.codacy.com/gh/BlackLightLabs/blacklight/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BlackLightLabs/blacklight&amp;utm_campaign=Badge_Grade)![PyPI - Downloads](https://img.shields.io/pypi/dm/blacklight?color=lime&label=Downloads%20from%20PyPi&logoColor=blue) [![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)

## Genetic algorithms in autoML. 
This project aims to use Genetic Algorithms to optimize the topologies of Deep Neural Networks (DNNs) and explore new possibilities that traditional optimization techniques might overlook. The fitness function of the algorithm is the accuracy of the model, and the genes represent the individual topologies.

## Installation 

Make sure you have Python "*" 
### Windows, Linux

1. Create new virtual environment:
   - ```pip install -m virtualenv```
   - ```python -m venv your_virtual_env_name```
   - ```source your_virtual_env_name\bin\activate```
2. Install the package:
   - ```pip install blacklight```
  
### For Maintainers
- Make sure you have [PDM](https://pdm-project.org/en/latest/) installed
- ```sh
  git clone https://github.com/BlackLightLabs/blacklight.git
  ```
- ```sh
  cd blacklight
  pdm install
  eval $(pdm venv activate)
  ```
- Happy Coding!
    
## Hypothesis

The hypothesis of this project is that DNN topologies will converge to either a local maximum or an absolute maximum over the evolution process, offering better performance than a DNN with randomly selected topology. For this experiment, the project will use equivalent activation functions (ReLU) and SGD for back-propagation, holding everything except the topology constant. Updated documentation coming soon.

## Methodology

The project utilizes a genetic algorithm to evolve the topology of the DNN. The algorithm starts with a randomly generated population of DNN topologies and evaluates their fitness using the accuracy of the model. The fittest individuals are selected for reproduction, while the weaker ones are discarded. The offspring of the selected individuals are then created through crossover and mutation. This process is repeated for a specified number of generations, and the best-performing topology is chosen as the final output.

## Documentation 
Documentation can be found at https://blacklightlabs.github.io/blacklight/html/index.html

## Cite

```bibtext
@software{github_blacklight_library,
title = {Blacklight},
author = {Cole Agard and Jackson Collins},
year = "2024",
url = "https://github.com/BlackLightLabs/blacklight"
}
```
