import tensorflow as tf


class BlacklightModel:
    def __init__(self, model_config, genes):
        self.model_history = None
        self.model_config = model_config
        self.genes = genes
        self.model = None

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.InputLayer(
                input_shape=self.model_config.get("input_shape")))
        for gene in self.genes:
            if gene[0] == "Conv2D":
                model.add(
                    tf.keras.layers.Conv2D(
                        gene[1],
                        gene[2],
                        activation=gene[3]))
            elif gene[0] == "MaxPooling2D":
                model.add(tf.keras.layers.MaxPooling2D(gene[1]))
            elif gene[0] == "Flatten":
                model.add(tf.keras.layers.Flatten())
            elif gene[0] == "Dense":
                model.add(tf.keras.layers.Dense(gene[1], activation=gene[2]))
            else:
                raise ValueError(f"Invalid gene type: {gene[0]}")

        target_layer = self.model_config.get("target_layer")
        model.add(
            tf.keras.layers.Dense(
                target_layer[0],
                activation=target_layer[1]))

        model.compile(
            optimizer=self.model_config.get("optimizer"),
            loss=self.model_config.get("loss"),
            metrics=self.model_config.get("metrics")
        )
        self.model = model

    def train_model(self, train_data):
        self.model_history = self.model.fit(
            x=train_data,
            epochs=self.model_config.get("epochs"),
            batch_size=self.model_config.get("batch_size"),
            validation_split=self.model_config.get("validation_split"),
            verbose=self.model_config.get("verbose"),
            class_weight=self.model_config.get("class_weight"),
            validation_data=self.model_config.get("validation_data"),
            callbacks=self.model_config.get("callbacks") if self.model_config.get("early_stopping") else None)

    def get_model(self):
        return self.model

    def get_model_history(self):
        return self.model_history

    def evaluate_model(self, train_data, test_data):
        # first train the model
        if self.model_history is None:
            self.train_model(train_data)
        # then evaluate the model
        results = self.model.evaluate(
            test_data,
            batch_size=self.model_config.get("batch_size"),
            verbose=self.model_config.get("verbose"),
            return_dict=True
        )
        # the fitness metric is extracted.
        fitness = results[self.model_config.get("fitness_metric")]
        return fitness
