"""
Utils for individuals.
"""

import random
from collections import OrderedDict


def get_min_length_chromosome(parents_chromosomes):
    """
    Get the chromosome with the minimum length.
    """
    if len(parents_chromosomes) == 0:
        raise ValueError("Chromosomes can not be empty.")

    return len(min([min(parental_chromosome.values(), key=len)
                    for parental_chromosome in parents_chromosomes], key=len))


def get_crossover_points_from_num_parents(num_parents, chromosome_length):
    """
    Get a list of crossover points for a given number of parents and chromosome length.
    """
    num_crossover_points = 0
    crossover_points = []
    index_chromosome_list = 1

    while num_crossover_points < num_parents - 1:
        index = random.randint(index_chromosome_list, chromosome_length - 1)
        crossover_points.append(index)
        index_chromosome_list = index
        num_crossover_points += 1

    return crossover_points


def merge_genes(parent_one_genes, parent_two_genes, points):
    """
    Merge two genes together.
    """

    def merge_genes_helper(dict_one, dict_two, base, link):
        """
        Given Parents Genes, merge the two dictionaries by link and base.
        :param parent_one_genes:
        :param parent_two_genes:
        :param base:
        :param link:
        :return:
        """
        result = OrderedDict()
        for key in base:
            result[key] = dict_one[key]
        for key in link:
            result[key] = dict_two[key]
        return result

    result = {}
    for gene_name in ["gene_0", "gene_1"]:
        p_one_genes = parent_one_genes[gene_name]
        p_two_genes = parent_two_genes[gene_name]
        base_one = list(p_one_genes.keys())[:points]
        link_one = list(p_two_genes.keys())[points:]

        base_two = list(p_two_genes.keys())[:points]
        link_two = list(p_one_genes.keys())[points:]

        result[gene_name + "_recombinant_one"] = merge_genes_helper(
            p_one_genes, p_two_genes, base_one, link_one)
        result[gene_name + "_recombinant_two"] = merge_genes_helper(
            p_two_genes, p_one_genes, base_two, link_two)
    return result


def mutate_dominant_gene(dominant_gene, MAX_LAYER_SIZE, **kwargs):
    """
    Mutate the dominant gene.
    """
    prob_of_mutation = 10 * kwargs.get("prob_of_mutation", 0.1)
    num_mutations_per_gene = kwargs.get(
        "num_mutations", len(list(dominant_gene.keys())) // 2)
    mutate = random.choices([True, False], weights=[
                            prob_of_mutation, 10 - prob_of_mutation], k=1)[0]
    if mutate:
        mutated_alleles = random.choices(
            list(
                dominant_gene.keys()), k=random.randint(
                1, num_mutations_per_gene))
        new_allele = random.randint(1, MAX_LAYER_SIZE)
        new_gene = [(new_allele, v) if k in mutated_alleles else (k, v)
                    for k, v in dominant_gene.items()]
        dom_gene = OrderedDict(new_gene)
        return dom_gene, mutate
    return dominant_gene, mutate
