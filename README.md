###### 
Blacklight  

[![test](https://github.com/BlackLightLabs/blacklight/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/BlackLightLabs/blacklight/actions/workflows/test.yml) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/449f7ff90fcb4340a4c90884d15f700a)](https://www.codacy.com/gh/BlackLightLabs/blacklight/dashboard?utm_source=github.com&utm_medium=referral&utm_content=BlackLightLabs/blacklight&utm_campaign=Badge_Coverage) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/449f7ff90fcb4340a4c90884d15f700a)](https://www.codacy.com/gh/BlackLightLabs/blacklight/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BlackLightLabs/blacklight&amp;utm_campaign=Badge_Grade)

# Genetic Deep Neural Net Topology Optimization
This project aims to use Genetic Algorithms to optimize the topologies of Deep Neural Networks (DNNs) and explore new possibilities that traditional optimization techniques might overlook. The fitness function of the algorithm is the accuracy of the model, and the genes represent the individual topologies.

## Author

This project is developed and maintained by Cole Agard, with help from Brandon Lee. 

## Hypothesis

The hypothesis of this project is that DNN topologies will converge to either a local maximum or an absolute maximum over the evolution process, offering better performance than a DNN with randomly selected topology. For this experiment, the project will use equivalent activation functions (ReLU) and SGD for back-propagation, holding everything except the topology constant.

## Methodology

The project utilizes a genetic algorithm to evolve the topology of the DNN. The algorithm starts with a randomly generated population of DNN topologies and evaluates their fitness using the accuracy of the model. The fittest individuals are selected for reproduction, while the weaker ones are discarded. The offspring of the selected individuals are then created through crossover and mutation. This process is repeated for a specified number of generations, and the best-performing topology is chosen as the final output.

## Documentation 
Documentation can be found at https://blacklightlabs.github.io/blacklight/html/index.html
