from individual import Individual
import random
class FeedForwardIndividual(Individual): 
    """
    Individual for a feed forward neural network.
    Description of genes: 
     - Dictionary of layers. 
        {number of layers: int, layer_activation: str}
    """

    def __init__(self, parents=None, population=None, MAX_NEURONS=64, MAX_LAYERS=10, NUM_PARENTS=2):
        super().__init__(parents, population)
        self.MAX_LAYERS = MAX_LAYERS
        self.MAX_NEURONS = MAX_NEURONS
        self.NUM_PARENTS = NUM_PARENTS

    def randomGenes(self):
        genes = [{random.randint(1, self.MAX_NEURONS): random.choice(["relu", "sigmoid", "tanh"]) 
                    for _ in range(self.MAX_LAYERS)} 
                    for _ in range(self.NUM_PARENTS)]

    def crossover(self):
        pass

    def mutate(self):
        pass

    def get_fitness(self):
        pass