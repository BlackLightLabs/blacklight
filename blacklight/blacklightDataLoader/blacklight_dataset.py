import numpy as np
import tensorflow as tf
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def parse_target_column(data):
    """
    Parse target column from data

    Parameters:
        - data: The input data to be classified. Must be of type pandas.DataFrame.

    Returns:
        - X: The feature columns of the input data.
        - y: The target column of the input data.
    """
    possible_prediction_columns = ["label", "prediction", "labels"]
    y_column_name = ""
    for column in data.columns:
        if str(column).lower() in possible_prediction_columns:
            y_column_name = column
            break

    y = data[y_column_name]
    X = data.to_numpy()[:, :-1]
    return X, y


def determine_input_data_type(X):
    """
    Determine the type of input data.
    Current supported datatypes are numpy.ndarray, pandas.DataFrame, tf.data.Dataset, a file location, or a directory.

    Parameters:
        - X: The input data to be classified. Must be of type numpy.ndarray, pandas.DataFrame, tf.data.Dataset, a file
             location, or a directory.
    """
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
        raise TypeError(
            "X must be of type numpy.ndarray, pandas.DataFrame, tf.data.Dataset, a file location, "
            "or a directory.")


def handle_pandas_data(X, y):
    """
    Handle pandas data by parsing the target column and encoding the target column if it is a string.

    Parameters:
        - X: The input data to be classified. Must be of type pandas.DataFrame.
        - y: The target column of the input data. Must be of type pandas.Series.
    """
    try:
        if y is None:
            X, y = parse_target_column(X)
        return np.array(X).astype('float32'), np.array(y)
    except BaseException:
        raise ValueError(
            "X (dataframe) could not be separated into label and feature columns.")


def handle_numpy_data(X, y):
    try:
        if y is None:
            X, y = X[:, :-1], X[:, -1]
        return X, y
    except IndexError:
        raise np.error_message(
            "X (np.array) could not be separated into label and feature columns.")


def handle_tf_data(X, y):
    return None, None


def handle_file_data(X):
    df = read_data_from_file(X, X.split(".")[-1])
    X, y = handle_pandas_data(df, None)
    return X, y


class BlacklightDataset(tf.keras.utils.Sequence):
    """
    A dataset for all blacklight populations to use. This class is a wrapper for tf.keras.utils.Sequence
    that handles all data types and formats, which also allows for multiprocessing during training.
    This class is used by the Individual class to extract data in the fit method for processing.

    Parameters:
        - X: The data to be used for training. This can be a numpy array, pandas dataframe, or a file location.
        - y: The labels for the data. This can be a numpy array, pandas series. If not provided, the last column of X will be used or the column labeled "label" or "labels".
        - batch_size: The size of the batch to be used for training. If not provided, the batch size will be the length of the data.
    """

    def __init__(self, X, y=None, batch_size=None):

        self.X = X
        self.y = y
        if y is None:
            type_of_data = determine_input_data_type(X)
            if type_of_data == "pandas":
                self.X, self.y = handle_pandas_data(self.X, self.y)
            elif type_of_data == "numpy":
                self.X, self.y = handle_numpy_data(self.X, self.y)
            elif type_of_data == "tf.data":
                self.X, self.y = handle_tf_data(self.X, self.y)
            elif type_of_data == "file":
                self.X, self.y = handle_file_data(self.X)
        self.batch_size = batch_size if batch_size else len(self.X)
        self.X_shape = self.X.shape[-1]
        # If the type of the target is string, encode it
        if isinstance(self.y[0], str):
            self.one_hot_encode_target()

    def one_hot_encode_target(self):
        y = LabelEncoder().fit_transform(self.y)
        self.y = to_categorical(y)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        """
        Grab a batch of data for use in training. If batch_size is not provided, the batch size will be the length of the data.

        Parameters:
            - idx: The index of the batch to be returned.
        """
        batch_x = self.X[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        return batch_x, batch_y


def read_data_from_file(data_location, data_type, **kwargs) -> pd.DataFrame:
    """
    Read data from a file location. This function will attempt to read the data as a CSV, JSON, Excel, or Parquet file.

    Parameters:
        - data_location: The location of the data to be read.
        - data_type: The type of data to be read. Must be one of 'csv', 'json', 'excel', or 'parquet'.
        - **kwargs: Keyword arguments to be passed to the read function.

    Returns:
        - read_data: The data read from the file location.
    """
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
