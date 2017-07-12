import pdb

"""A generic module to read data."""
""" https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477 """

import numpy
import collections
from tensorflow.python.framework import dtypes

import sys 
sys.path.append('..')
import trains_dataset

from datetime import datetime

class DataSet(object):
    """Dataset class object."""

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=True):
        """Initialize the class."""
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                images.shape[1] * images.shape[2])

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, print_epochs=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            if(print_epochs):
                print str(datetime.now()) + ": Epoch completed " + str(self._epochs_completed)
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._epochs_completed


def read_data_sets(train_dir, fake_data=False, one_hot=False,
                        dtype=dtypes.float64, reshape=True,
                        validation_size=5000):
    # pdb.set_trace()
    training_data, test_data = trains_dataset.load_trains_dataset()

    training_images = []
    training_labels = []
    for training_tuple in training_data:
      image = training_tuple[0].reshape(training_tuple[0].shape[0])
      label = training_tuple[1].reshape(training_tuple[1].shape[0])
      training_images.append(image)
      training_labels.append(label)

    training_images = numpy.asarray(training_images)
    training_labels = numpy.asarray(training_labels)

    test_images = []
    test_labels = []
    for test_tuple in test_data:
      image = test_tuple[0].reshape(test_tuple[0].shape[0])
      label = trains_dataset.vectorized_output(test_tuple[1])
      label = label.reshape(label.shape[0])
      test_images.append(image)
      test_labels.append(label)

    test_images = numpy.asarray(test_images)
    test_labels = numpy.asarray(test_labels)

    train = DataSet(training_images, training_labels, dtype=dtype, reshape=reshape)
    # validation = DataSet(validation_images, validation_labels, dtype=dtype,
    #     reshape=reshape)

    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    ds = collections.namedtuple('Datasets', ['train', 'test'])

    return ds(train=train, test=test)



def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot