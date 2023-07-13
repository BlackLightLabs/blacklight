import numpy as np
import pytest
import tensorflow as tf

from blacklight.engine.utils import data_utils


def test_split_data_has_one_batch_error():
    with pytest.raises(ValueError) as info:
        data_utils.split_dataset(
            tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3])).batch(12),
            0.2,
        )

    assert "The dataset should at least contain 2 batches" in str(info.value)


def test_unzip_dataset_doesnt_unzip_single_dataset():
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 32, 2))
    dataset = data_utils.unzip_dataset(dataset)[0]
    dataset = data_utils.unzip_dataset(dataset)[0]
    assert data_utils.dataset_shape(dataset).as_list() == [32, 2]


def test_cast_to_string_with_float32():
    tensor = tf.constant([0.1, 0.2], dtype=tf.float32)
    assert tf.string == data_utils.cast_to_string(tensor).dtype


def test_cast_to_float32_from_float32():
    tensor = tf.constant([0.1, 0.2], dtype=tf.float32)
    assert tf.float32 == data_utils.cast_to_float32(tensor).dtype


def test_cast_to_float32_from_string():
    tensor = tf.constant(["0.3"], dtype=tf.string)
    assert tf.float32 == data_utils.cast_to_float32(tensor).dtype
