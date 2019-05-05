# Leonard R. Kosta Jr.

from keras.datasets import imdb
from keras import models
from keras import layers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)
