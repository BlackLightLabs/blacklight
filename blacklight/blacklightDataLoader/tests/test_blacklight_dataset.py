from blacklight.blacklightDataLoader import BlacklightDataset
from blacklight.blacklightDataLoader.blacklight_dataset import read_data_from_file, handle_pandas_data, \
    handle_numpy_data, handle_file_data, parse_target_column, determine_input_data_type
import pandas as pd
import numpy as np
import pytest


def test_parse_target_column():
    test_data = pd.read_csv(
        "blacklight/blacklightDataLoader/tests/data/Iris.csv")
    X, y = parse_target_column(test_data)
    assert X.shape == (150, 5)
    assert y.shape == (150, 3)
    assert isinstance(X, np.ndarray)


def test_read_error_type():
    with pytest.raises(ValueError):
        read_data_from_file(
            "blacklight/blacklightDataLoader/tests/data/Iris.csv",
            "otherdatatype")


def test_read_data_from_file():
    data = read_data_from_file(
        "blacklight/blacklightDataLoader/tests/data/Iris.csv", "csv")
    assert data.shape == (150, 6)
    data = read_data_from_file(
        "blacklight/blacklightDataLoader/tests/data/Iris.json", "json")
    assert data.shape == (150, 6)
    data = read_data_from_file(
        "blacklight/blacklightDataLoader/tests/data/Iris.parquet",
        "parquet")
    assert data.shape == (150, 6)
    data = read_data_from_file(
        "blacklight/blacklightDataLoader/tests/data/Iris.xlsx", "excel")
    assert data.shape == (150, 6)


def test_handle_pandas_data():
    test_data = pd.read_csv(
        "blacklight/blacklightDataLoader/tests/data/Iris.csv")
    X, y = handle_pandas_data(test_data, None)
    assert X.shape == (150, 5)
    assert y.shape == (150,)
    assert isinstance(y, np.ndarray)
    assert isinstance(X, np.ndarray)


def test_handle_numpy_data():
    test_data = pd.read_csv(
        "blacklight/blacklightDataLoader/tests/data/Iris.csv")
    X, y = handle_numpy_data(test_data.to_numpy(), None)
    assert X.shape == (150, 5)
    assert y.shape == (150,)
    assert isinstance(y, np.ndarray)
    assert isinstance(X, np.ndarray)


def test_handle_file_data():
    X, y = handle_file_data(
        "blacklight/blacklightDataLoader/tests/data/Iris.csv")
    assert X.shape == (150, 5)
    assert y.shape == (150,)
    assert isinstance(y, np.ndarray)
    assert isinstance(X, np.ndarray)


def test_determine_input_data_type():
    X_df = pd.read_csv("blacklight/blacklightDataLoader/tests/data/Iris.csv")
    X_numpy = pd.read_csv(
        "blacklight/blacklightDataLoader/tests/data/Iris.csv").to_numpy()
    X_file = "blacklight/blacklightDataLoader/tests/data/Iris.csv"
    X_directory = "blacklight/blacklightDataLoader/tests/data/"

    assert determine_input_data_type(X_df) == "pandas"
    assert determine_input_data_type(X_numpy) == "numpy"
    assert determine_input_data_type(X_file) == "file"
    assert determine_input_data_type(X_directory) == "directory"

    with pytest.raises(TypeError):
        determine_input_data_type(1)


def test_Blacklight_Dataset():
    X_df = pd.read_csv("blacklight/blacklightDataLoader/tests/data/Iris.csv")
    X_numpy = pd.read_csv(
        "blacklight/blacklightDataLoader/tests/data/Iris.csv").to_numpy()
    X_file = "blacklight/blacklightDataLoader/tests/data/Iris.csv"

    dataset_from_df = BlacklightDataset(X_df)
    dataset_from_numpy = BlacklightDataset(X_numpy)
    dataset_from_file = BlacklightDataset(X_file)

    assert dataset_from_df.X.shape == (150, 5)
    assert dataset_from_df.y.shape == (150,)
    assert isinstance(dataset_from_df.y, np.ndarray)
    assert isinstance(dataset_from_df.X, np.ndarray)

    assert dataset_from_numpy.X.shape == (150, 5)
    assert dataset_from_numpy.y.shape == (150,)
    assert isinstance(dataset_from_numpy.y, np.ndarray)
    assert isinstance(dataset_from_numpy.X, np.ndarray)

    assert dataset_from_file.X.shape == (150, 5)
    assert dataset_from_file.y.shape == (150,)
    assert isinstance(dataset_from_file.y, np.ndarray)
    assert isinstance(dataset_from_file.X, np.ndarray)

    assert dataset_from_df.__getitem__(0)[0].shape == (150, 5)
    assert dataset_from_df.__getitem__(0)[1].shape == (150,)
