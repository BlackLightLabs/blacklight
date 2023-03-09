import numpy as np
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


def parse_target_column(data):
    """
    Parse target column from data
    :param data: pd.DataFrame
    :return: pd.DataFrame, pd.Series
    """
    possible_prediction_columns = ["label", "prediction", "labels"]
    y_column_name = ""
    for column in data.columns:
        if str(column).lower() in possible_prediction_columns:
            y_column_name = column
            break

    y = data.pop(y_column_name)
    X = data
    return X, y


def determine_input_data_type(X):
    if isinstance(X, pd.DataFrame):
        return "pandas"
    elif isinstance(X, np.ndarray):
        return "numpy"
    elif isinstance(X, tf.data.Dataset):
        return "tf.data"
    elif isinstance(X, str):
        if "." in X:
            return "file"
        else:
            return "directory"
    else:
        raise TypeError("X must be of type numpy.ndarray, pandas.DataFrame, tf.data.Dataset, a file location, "
                        "or a directory.")


def handle_pandas_data(X, y):
    try:
        if y is None:
            X, y = parse_target_column(X)
        if isinstance(y[0], str):
            y = LabelEncoder().fit_transform(y)
        return X, y
    except:
        raise ValueError("X (dataframe) could not be separated into label and feature columns.")


def handle_numpy_data(X, y):
    try:
        if y is None:
            X, y = X[:, :-1], X[:, -1]
        if isinstance(y[0], str):
            y = LabelEncoder().fit_transform(y)
        return X, y
    except IndexError:
        raise np.error_message("X (np.array) could not be separated into label and feature columns.")


def handle_tf_data(X, y):
    return None, None


def handle_file_data(X):
    df = read_data_from_file(X, X.split(".")[-1])
    X, y = handle_pandas_data(df, None)
    return X, y


class BlacklightDataset(tf.keras.utils.Sequence):
    def __init__(self, X, y=None, batch_size=None):
        type_of_data = determine_input_data_type(X)
        self.X = X
        self.y = y
        if type_of_data == "pandas":
            self.X, self.y = handle_pandas_data(self.X, self.y)
        elif type_of_data == "numpy":
            self.X, self.y = handle_numpy_data(self.X, self.y)
        elif type_of_data == "tf.data":
            self.X, self.y = handle_tf_data(self.X, self.y)
        elif type_of_data == "file":
            self.X, self.y = handle_file_data(self.X)

        self.batch_size = batch_size if batch_size else len(self.x)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        return batch_x, batch_y


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        if isinstance(y_set[0], str):
            self.y = LabelEncoder().fit_transform(self.y)
            self.num_classes = len(set(self.y))
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


def read_data_from_file(data_location, data_type, **kwargs) -> pd.DataFrame:
    read_data = None
    try:
        # Attempt to read CSV Data
        if data_type == "csv":
            read_data = pd.read_csv(data_location, **kwargs)
        # Attempt to read JSON Data
        elif data_type == "json":
            read_data = pd.read_json(data_location, **kwargs)
        # Attempt to read Excel Data
        elif data_type == "excel":
            read_data = pd.read_excel(data_location, **kwargs)
        # Attempt to read Parquet Data
        elif data_type == "parquet":
            read_data = pd.read_parquet(data_location, **kwargs)
        # Raise ValueError if data_type is not one of the above
        else:
            raise ValueError(
                "data_type must be one of 'csv', 'json', 'excel', or 'parquet'")
    # Raise FileNotFoundError if data_location is not found in any of above
    # file types
    except FileNotFoundError:
        raise FileNotFoundError(f"{data_type} file not found at data_location")

    return pd.DataFrame(read_data)
