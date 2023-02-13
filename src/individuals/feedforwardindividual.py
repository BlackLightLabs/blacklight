from src.individuals.individual import Individual
from src.individuals.individualutils import *
import random

class FeedForwardIndividual(Individual): 
    """
    Individual for a feed forward neural network.
    Description of genes: 
     - Dictionary of layers. 
        {number of layers: int, layer_activation: str}
    """

    def __init__(self, parents_genes=None, population=None, MAX_NEURONS=64, MAX_LAYERS=10, NUM_PARENTS=2, **kwargs):
        super().__init__(parents_genes, population)
        # Feed Forward Neural Network parameters
        self.MAX_LAYERS = MAX_LAYERS
        self.MAX_NEURONS = MAX_NEURONS
        self.NUM_PARENTS = NUM_PARENTS
        # Keras parameters -> Maybe these need to be in the population, not the individual?
        self.EPOCHS = kwargs.get("EPOCHS", 1000)
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 2048)
        self.VERBOSE = kwargs.get("VERBOSE", 0)
        self.EARLY_STOPPING = kwargs.get("EARLY_STOPPING", True)
        self.VALIDATION_SPLIT = kwargs.get("VALIDATION_SPLIT", 0.2)
        self.genes = self._random_genes() if self.need_new_genes else self._crossover()
    def _random_genes(self):
        """
        Randomly initialize genes for each type of individual if there are no parents.
        Feed Forward Neural Networks are made up of layers. Each layer has a number of neurons and an activation function.
        Initialize genes as a list of dictionaries. Each dictionary represents a layer.
        """
        genes = {f"gene_{i}": {random.randint(1, self.MAX_NEURONS): random.choice(["relu", "sigmoid", "tanh"])
                    for _ in range(self.MAX_LAYERS)} 
                    for i in range(2)}
        return genes

    def _crossover(self):
        """
        ########################################################
         Chromasome 1   |     Chromasome 2   |     Chromasome 3
        ########################################################
        {12: 'softmax', |     {8: 'relu',    |    {8: 'relu',
         4: 'softmax',  |      2: 'relu',    |     2: 'relu',
         7: 'relu',     |      4: 'relu'}    |     4: 'softmax'}
         4: 'relu',     |      7: 'relu',    |
         7: 'Conv2D',   |      4: 'softmax'} |
         12: 'relu',    |                    |
         4: 'relu'}     |                    |
                        |                    |
         #######################################################
         #
         #                        CROSS
         #                         OVER
         #                  Random Choice: element 3.
         #######################################################
         #         Recombinant 1    |        Recombinant 2
         #######################################################
                {8: 'relu',         |     {12: 'softmax',
                 2: 'relu',         |      4: 'softmax',
                 4: 'relu',         |      7: 'relu',
                 4: 'relu',         |      7: 'relu',
                 7: 'Conv2D'        |      4: 'softmax'}
                 12: 'relu'         |
                 4: 'relu'}         |

        TODO:
            Per INDIVIDUAL
                Chromasome -> 1
                Chromasome -> 2
                Chromasome -> 3
            P: We need to select one of the three Chromosomes gets RUN by the brain (population)
            A: in INDIVIDUAL
                    self.dominantChromisome = choice of the three carried
                    self.chromasomeList = [Chromasome 1, Chromasome 2, Chromasome 3]
                    Need to define:
                        List of predetermined structures that perform well for certain tasks:
                            EX, ConvDNET2D -> images (copy structure, compare similarity to all three chromasomes
                                Collapse dictionaries into a list, and compare to database of chromasomes
                    At end of simulation, add the best performer's chromosome to the dataset.
                        "Phenotype expression"
        :return:
        """
        # Get crossover points
        smallest_parental_chromosome = get_min_length_chromosome(self.parents_genes)
        crossover_points = get_crossover_points_from_num_parents(self.NUM_PARENTS, len(smallest_parental_chromosome)) 

        # Define recombined genes
        recombinants = []

        pass

    def _mutate(self):
        pass

    def make_model(self):
        pass

    def get_fitness(self):
        pass
