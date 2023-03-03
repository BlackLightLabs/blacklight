import numpy as np
from blacklight.dataLoaders.utils import read_data_from_file
import tensorflow as tf
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """
    An abstract class to hold general methods between dataloader instances
    and provide a structure to satisfy Liwosky Substitution.
    """

    def __init__(self, data):
        """
        Specify data_location, which may have many types:
            (PostgresSQL, SQLite, Dataframe, Redis, etc...)

        :param args: Location of data source, with other args depending on source.
        """
        # Must be of time pd.DataFrame
        self.data = data
        pass

    def extractData(self):
        """
        Expects self.data to be of type pd.DataFrame
        :type pd.DataFrame
        """
        possible_prediction_columns = ["label", "prediction", "labels"]
        y_column_name = ""
        for column in self.data.columns:
            if str(column).lower() in possible_prediction_columns:
                y_column_name = column
                break

        y = np.array(self.data.pop(y_column_name))
        X = np.array(self.data)
        return X, y


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        if isinstance(y_set[0], str):
            self.y = LabelEncoder().fit_transform(self.y)
        self.batch_size = batch_size if batch_size else len(self.x)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        return batch_x, batch_y


class FileDataLoader(DataLoader):
    """
    Take data from a file
    """

    def __init__(self, data_location, data_type):
        self.data = read_data_from_file(data_location, data_type)
        super().__init__(self.data)
        self.X, self.y = self.extractData()

    def get_dataset(self, batch_size):
        return Dataset(self.X, self.y, batch_size)


class DFDataLoader(DataLoader):
    """
    Take data from predefined DataFrame
    """

    def __init__(self, data):
        self.data = data
        super().__init__(self.data)
        self.X, self.y = self.extractData()

    def get_dataset(self, batch_size):
        return Dataset(self.X, self.y, batch_size)


def choose_data_loader(data_location):
    """
    Choose the correct data loader based on the data type.
    """
    if isinstance(data_location, str):
        return FileDataLoader(data_location, data_location.split(".")[-1])
    elif isinstance(data_location, pd.DataFrame):
        return DFDataLoader(data_location)
    else:
        raise ValueError("Data type not recognized.")
