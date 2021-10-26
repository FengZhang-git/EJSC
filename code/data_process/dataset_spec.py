# coding=utf-8
# Copyright 2021 The Meta-Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Interfaces for dataset specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

from absl import logging
# from meta_dataset import data
# from meta_dataset.data import imagenet_specification
import data_process.learning_spec as learning_spec
import numpy as np
import six
from six.moves import cPickle as pkl
from six.moves import range
from six.moves import zip


def get_total_images_per_class(data_spec, class_id=None, pool=None):
  """Returns the total number of images of a class in a data_spec and pool.

  Args:
    data_spec: A DatasetSpecification, or BiLevelDatasetSpecification.
    class_id: The class whose number of images will be returned. If this is
      None, it is assumed that the dataset has the same number of images for
      each class.
    pool: A string ('train' or 'test', optional) indicating which example-level
      split to select, if the current dataset has them.

  Raises:
    ValueError: when
      - no class_id specified and yet there is class imbalance, or
      - no pool specified when there are example-level splits, or
      - pool is specified but there are no example-level splits, or
      - incorrect value for pool.
    RuntimeError: the DatasetSpecification is out of date (missing info).
  """
  if class_id is None:
    if len(set(data_spec.images_per_class.values())) != 1:
      raise ValueError('Not specifying class_id is okay only when all classes'
                       ' have the same number of images')
    class_id = 0

  if class_id not in data_spec.images_per_class:
    raise RuntimeError('The DatasetSpecification should be regenerated, as '
                       'it does not have a non-default value for class_id {} '
                       'in images_per_class.'.format(class_id))
  num_images = data_spec.images_per_class[class_id]

  if pool is None:
    if isinstance(num_images, collections.Mapping):
      raise ValueError('DatasetSpecification {} has example-level splits, so '
                       'the "pool" argument has to be set (to "train" or '
                       '"test".'.format(data_spec.name))
  elif not data.POOL_SUPPORTED:
    raise NotImplementedError('Example-level splits or pools not supported.')

  return num_images



class DatasetSpecification(
    collections.namedtuple('DatasetSpecification',
                           ('name, images_per_class, '
                            'class_names, path, file_pattern'))):
  """The specification of a dataset.

    Args:
      name: string, the name of the dataset.
      classes_per_split: a dict specifying the number of classes allocated to
        each split.
      images_per_class: a dict mapping each class id to its number of images.
        Usually, the number of images is an integer, but if the dataset has
        'train' and 'test' example-level splits (or "pools"), then it is a dict
        mapping a string (the pool) to an integer indicating how many examples
        are in that pool. E.g., the number of images could be {'train': 5923,
        'test': 980}.
      class_names: a dict mapping each class id to the corresponding class name.
      path: the path to the dataset's files.
      file_pattern: a string representing the naming pattern for each class's
        file. This string should be either '{}.tfrecords' or '{}_{}.tfrecords'.
        The first gap will be replaced by the class id in both cases, while in
        the latter case the second gap will be replaced with by a shard index,
        or one of 'train', 'valid' or 'test'. This offers support for multiple
        shards of a class' images if a class is too large, that will be merged
        later into a big pool for sampling, as well as different splits that
        will be treated as disjoint pools for sampling the support versus query
        examples of an episode.
  """

  def initialize(self, restricted_classes_per_split=None):
    """Initializes a DatasetSpecification.

    Args:
      restricted_classes_per_split: A dict that specifies for each split, a
        number to restrict its classes to. This number must be no greater than
        the total number of classes of that split. By default this is None and
        no restrictions are applied (all classes are used).

    Raises:
      ValueError: Invalid file_pattern provided.
    """
    # Check that the file_pattern adheres to one of the allowable forms
    if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
      raise ValueError('file_pattern must be either "{}.tfrecords" or '
                       '"{}_{}.tfrecords" to support shards or splits.')
    

  def get_total_images_per_class(self, class_id=None, pool=None):
    """Returns the total number of images for the specified class.

    Args:
      class_id: The class whose number of images will be returned. If this is
        None, it is assumed that the dataset has the same number of images for
        each class.
      pool: A string ('train' or 'test', optional) indicating which
        example-level split to select, if the current dataset has them.

    Raises:
      ValueError: when
        - no class_id specified and yet there is class imbalance, or
        - no pool specified when there are example-level splits, or
        - pool is specified but there are no example-level splits, or
        - incorrect value for pool.
      RuntimeError: the DatasetSpecification is out of date (missing info).
    """
    return get_total_images_per_class(self, class_id, pool=pool)

  def get_classes(self, split):
    """Gets the sequence of class labels for a split.

    Labels are returned ordered and without gaps.

    Args:
      split: A Split, the split for which to get classes.

    Returns:
      The sequence of classes for the split.

    Raises:
      ValueError: An invalid split was specified.
    """
    return range(len(self.class_names.keys()))
    # return get_classes(split, self.classes_per_split)

  def to_dict(self):
    """Returns a dictionary for serialization to JSON.

    Each member is converted to an elementary type that can be serialized to
    JSON readily.
    """
    # Start with the dict representation of the namedtuple
    ret_dict = self._asdict()
    # Add the class name for reconstruction when deserialized
    ret_dict['__class__'] = self.__class__.__name__
    # Convert Split enum instances to their name (string)
    ret_dict['classes_per_split'] = {
        split.name: count
        for split, count in six.iteritems(ret_dict['classes_per_split'])
    }
    # Convert binary class names to unicode strings if necessary
    class_names = {}
    for class_id, name in six.iteritems(ret_dict['class_names']):
      if isinstance(name, six.binary_type):
        name = name.decode()
      elif isinstance(name, np.integer):
        name = six.text_type(name)
      class_names[class_id] = name
    ret_dict['class_names'] = class_names
    return ret_dict


  

