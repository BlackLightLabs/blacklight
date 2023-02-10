import pandas as pd
import numpy as np
from blacklight.dataLoaders.utils import read_data_from_file

from abc import ABC, abstractmethod


class DataLoader:
    """
    An abstract class to hold general methods between dataloader instances
    and provide a structure to satisfy Liwosky Substitution.
    """

    def __init__(self, *args):
        """
        Specify data_location, which may have many types:
            (PostgresSQL, SQLite, Dataframe, Redis, etc...)

        :param args: Location of data source, with other args depending on source.
        """
        self.data = args[0]
        self.dataType = args[1]
        pass


class FileDataLoader(DataLoader):
    """
    Take data from predefined DataFrame
    """

    def __init__(self, data_location, data_type, **kwargs):
        self.data = read_data_from_file(data_location, data_type, **kwargs)
        self.dataType = data_type

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
