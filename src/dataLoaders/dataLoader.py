import pandas as pd
import numpy as np
from src.dataLoaders.utils import read_data_from_file

from abc import ABC, abstractmethod


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


class FileDataLoader(DataLoader):
    """
    Take data from a file
    """

    def __init__(self, data_location, data_type):
        self.data = read_data_from_file(data_location, data_type, **kwargs)
        super().__init__(self.data)
        self.X, self.y = self.extractData()


class DFDataLoader(DataLoader):
    """
    Take data from predefined DataFrame
    """

    def __init__(self, data):
        self.data = data
        super().__init__(self.data)
        self.X, self.y = self.extractData()


def choose_data_loader(data_location, data_type):
    """
    Choose the correct data loader based on the data type.
    """
    if data_type == "file":
        return FileDataLoader(data_location)
    elif data_type == "df":
        return DFDataLoader(data_location)
    else:
        raise ValueError("Data type not recognized.")
