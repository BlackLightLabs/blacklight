from blacklight.autoML._feed_forward import FeedForward
import pandas as pd
import pytest


@pytest.mark.slow
def test_end_to_end_feed_forward_NN():
    data = pd.read_csv('blacklight/autoML/tests/data/Iris.csv', index_col=0)
    ff_autoML = FeedForward(4, 2, 0.2, 2)
    # Fit the model
    ff_autoML.fit(data)
    # Get model
    ff_autoML.print_model_summary()
    ff_autoML.print_model_training_history()
