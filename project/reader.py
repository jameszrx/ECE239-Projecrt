from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random
import params
from six.moves import xrange

import numpy as np
import h5py

from tensorflow.python.platform import gfile


class Reader(object):
    def __init__(self, data_dir):
        search_path = os.path.join(data_dir, '*.mat')
        features = np.zeros([288 * 9, params.LENGTH, params.FEATURE_DIMENSION])
        labels = np.zeros([288 * 9])
        mat_count = 0
        # convert [769, 773) to [0, 4)
        converter = np.vectorize(lambda t: t - 769)
        for mat_path in gfile.Glob(search_path):
            mat = h5py.File(mat_path, 'r')
            x = np.copy(mat['image'])
            x = x[:, 0:params.FEATURE_DIMENSION, :]
            x = np.transpose(x, (0, 2, 1))
            x = np.asarray(x, dtype=np.float32)
            x = np.nan_to_num(x)
            y = np.copy(mat['type'])
            y = y[0, 0:x.shape[0]:1]
            y = np.asarray(y, dtype=np.int64)
            y = converter(y)

            features[288 * mat_count:288 * (mat_count + 1), :, :] = x
            labels[288 * mat_count:288 * (mat_count + 1)] = y
            mat_count += 1

        # shuffle
        num_trials = labels.shape[0]
        random.seed(params.RANDOM_SEED)
        index_shuffle = list(range(num_trials))
        random.shuffle(index_shuffle)
        features_shuffle = np.asarray([features[i] for i in index_shuffle])
        labels_shuffle = np.asarray([labels[i] for i in index_shuffle])

        training_index = int(np.floor(num_trials * (1 - params.VALIDATION_PERCENTAGE - params.TESTING_PERCENTAGE)))
        validation_index = training_index + int(np.floor(num_trials * params.VALIDATION_PERCENTAGE))
        test_index = num_trials

        self.features = {'training': features_shuffle[0:training_index],
                         'validation': features_shuffle[training_index:validation_index],
                         'testing': features_shuffle[validation_index:test_index]}
        self.labels = {'training': labels_shuffle[:training_index],
                       'validation': labels_shuffle[training_index:validation_index],
                       'testing': labels_shuffle[validation_index:test_index]}

    def get_data(self, how_many, offset, mode):
        candidates = list(range(len(self.labels[mode])))
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        features = np.zeros((sample_count, params.LENGTH, params.FEATURE_DIMENSION))
        labels = np.zeros(sample_count)

        pick_deterministically = (mode != 'training')

        for i in xrange(offset, offset + sample_count):
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            feature = self.features[mode][sample_index, :, :]
            label = self.labels[mode][sample_index]

            features[i - offset] = feature
            labels[i - offset] = label

        return features, labels
