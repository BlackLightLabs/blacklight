import tensorflow as tf
from tensorflow import keras
from blacklight.base.individual import Individual
from blacklight.base.chromosomes.feed_forward_chromosome import FeedForwardChromosome, \
    handle_feed_forward_chromosome_cross_over

from dataclasses import dataclass


def parse_keras_options(options):
    """
    Parse keras options.
    """
    train_options = {}
    model_options = {}

    train_options["epochs"] = options.get("epochs", 1000)
    train_options["batch_size"] = options.get("batch_size", 2048)
    train_options["verbose"] = options.get("verbose", 0)
    train_options["validation_data"] = options.get("validation_data", None)
    train_options["early_stopping"] = options.get("early_stopping", True)
    train_options["output_bias"] = options.get("output_bias", True)
    train_options["callbacks"] = options.get(
        "callbacks", FeedForwardConstants.default_callbacks)
    train_options["use_multiprocessing"] = options.get(
        "use_multiprocessing", True)

    model_options["loss"] = options.get("loss", 'categorical_crossentropy')
    model_options["metrics"] = options.get(
        "metrics", FeedForwardConstants.default_metrics)
    model_options["learning_rate"] = options.get("learning_rate", .0001)
    model_options["optimizer"] = options.get(
        "optimizer", tf.keras.optimizers.Adam)

    return train_options, model_options


class FeedForwardIndividual(Individual):
    """
    Individual for a feed forward neural network.

    Parameters:
            parents_genes: list of genes from parents
            population: population object that this individual belongs to
            MAX_NEURONS: maximum number of neurons allowed
            MAX_LAYERS: maximum number of layers allowed
            NUM_PARENTS: number of parents used for crossover
            **kwargs: arguments to be passed to Keras

    Returns:
        None
    """

    def __init__(
            self,
            parents_genes=None,
            population=None,
            MAX_NEURONS=64,
            MAX_LAYERS=10,
            **kwargs):
        """
        Initializes a FeedForwardIndividual.
        """
        super().__init__(parents_genes, population)
        # Feed Forward Neural Network parameters
        self.MAX_LAYERS = MAX_LAYERS
        self.MAX_NEURONS = MAX_NEURONS
        # Keras parameters -> Maybe these need to be in the population, not the
        # individual?
        options = kwargs
        self.train_options, self.model_options = parse_keras_options(options)
        # Class parameters
        self.dominant_gene = None

        self.model = None
        self.fitness_metric = kwargs.get("fitness_metric", 'auc')
        # data
        self.train_data = population.get_training_data()
        self.test_data = population.get_testing_data()
        self.X_shape = self.train_data.X_shape
        if self.need_new_genes:
            self.chromosome = FeedForwardChromosome(
                input_shape=self.X_shape,
                num_classes=self.train_data.num_classes(),
                genes=None,
                mutation_prob=0.1,
                model_params=self.model_options)
        else:
            self.chromosome = self._crossover(self.parents_genes)

    def _crossover(self, parents_chromosomes):
        parent_one_chrome, parent_two_chrome = parents_chromosomes
        return handle_feed_forward_chromosome_cross_over(
            parent_one_chrome, parent_two_chrome)

    def get_fitness(self):
        """
        Get the fitness of the individual.
        """
        # Evaluate the model
        fitness = self._evaluate_model()
        self.fitness = fitness
        return fitness

    def mate(self, other_individual):
        """
        Mate with another individual.
        :param other_individual: Individual to mate with.
        :return:
        """
        # Get genes from parents
        parents_genes = (self.chromosome, other_individual.chromosome)
        # Create child
        child = FeedForwardIndividual(
            parents_genes=parents_genes,
            population=self.population)
        # Return child
        return child

    def _train_model(self):
        """
        Train the model.
        :return:
        """
        # Train the model
        self.model.fit(
            x=self.train_data,
            epochs=self.train_options["epochs"],
            batch_size=self.train_options["batch_size"],
            verbose=self.train_options["verbose"],
            class_weight=self.train_options["class_weight"],
            validation_data=self.train_options["validation_data"],
            use_multiprocessing=self.train_options["use_multiprocessing"],
            callbacks=self.train_options['callbacks'] if self.train_options['early_stopping'] else None)

    def _evaluate_model(self):
        """
        Evaluate the model.
        """
        # First make the model
        self.model = self.chromosome.get_model()
        # Then train the model
        self._train_model()
        # Then get the fitness
        results = self.model.evaluate(
            self.test_data,
            batch_size=self.train_options["batch_size"],
            verbose=self.train_options["verbose"],
            return_dict=True
        )
        # the fitness metric is extracted.
        fitness = results[self.fitness_metric]
        return fitness


@dataclass
class FeedForwardConstants:
    """Constants for FeedForward NeuralNetorks"""

    default_metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.CategoricalAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    default_callbacks = tf.keras.callbacks.EarlyStopping(
        monitor='auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)
