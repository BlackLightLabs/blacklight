import blacklight.engine.model_options as mo
import blacklight.engine.model_creator as mc


class Chromosome:
    def __init__(self, model_params: mo.ModelConfig, genes: list[tuple[object, *tuple[int, ...], object]] | None = None, mutation_prob: float | None = None, random_genes: bool = True):
        self.model_params = model_params

        has_new_genes = genes is not None

        self.mutation_prob = mutation_prob

        # self.genes = genes if genes else self._random_genes()
        if genes:
            self.genes = genes
        elif random_genes:
            self.genes = self._random_genes()
        else:
            # TODO: implement smarter system for initializing new genes
            self._random_genes()
            print("Creating random gene despite random_genes being false, this will be fixed later")

        self.length = len(self.genes) # pyright: ignore [reportArgumentType]
        
        if has_new_genes:
            self._mutate()
        
        # constructor creates model
        self.model = mc.BlacklightModel(self.model_params, self.genes) # pyright: ignore [reportArgumentType]

    def _random_genes(self):
       pass

    @staticmethod
    def crossover(chromosome0: object, chromosome1: object) -> object:
        pass

    def _mutate(self):
        pass

    def order_chromosomes(self, chromosome0: 'Chromosome', chromosome1: 'Chromosome') -> tuple['Chromosome', 'Chromosome']:
        if chromosome0.length > chromosome1.length:
            return chromosome1, chromosome0
        else:
            return chromosome0, chromosome1
