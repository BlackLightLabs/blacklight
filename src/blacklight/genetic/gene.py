from typing import Any # pyright: ignore [reportAny]
import torch

# class GeneType(Enum):
#     LINEAR      = 0
#     CONV        = 1
#     POOLING     = 2
#     PADDING     = 4
#     NORM        = 4
#     RECURRENT   = 5
#     DROPOUT     = 6
#     SPARSE      = 7
#     VISION      = 8
#     SHUFFLE     = 9
#     ACTIVATION  = 10

# represent model as a list of Gene objects
# the functionality implemented within the gene should by the laws of nature
# naturally scale into the chromosome without force
# Chromosome will inherit Gene
class Gene:
    def __init__(self, gene_type: object=None, dna: list[Any] | None=None):
        self.gene_type = gene_type if gene_type else torch.nn.Linear
        self.dna = dna if dna else []

    def crossover(self, other: 'Gene'):
        if self.gene_type == other.gene_type:
            
        else:
            # TODO: Implement gene repair
            print("Gene types must be the same:", self.gene_type, "is not equal to", other.gene_type)

    def mutate(self):
        pass

    # return the types of layers that this layer can be proceeded by list[layerType, ...]
    # return empty list if it can be proceeded by any
    def rule(self):
        pass
