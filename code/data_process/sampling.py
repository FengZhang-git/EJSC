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
"""Sampling the composition of episodes.

The composition of episodes consists in the number of classes (num_ways), which
classes (relative class_ids), and how many examples per class (num_support,
num_query).

This module aims at replacing `sampler.py` in the new data pipeline.
"""
# TODO(lamblinp): Update variable names to be more consistent
# - target, class_idx, label

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import data_process.dataset_spec as dataset_spec_lib
# import imagenet_specification
import numpy as np
from six.moves import zip


logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='new.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

# Module-level random number generator. Initialized randomly, can be seeded.
RNG = np.random.RandomState(seed=None)

# How the value of MAX_SPANNING_LEAVES_ELIGIBLE was selected.
# This controls the upper bound on the number of leaves that an internal node
# may span in order for it to be eligible for selection. We found that this
# value is the minimum such value that allows every leaf to be reachable. By
# decreasing it, not all leaves would be reachable (therefore some classes would
# never be used). By increasing it, all leaves would still be reachable but we
# would sacrifice naturalness more than necessary (since when we sample an
# internal node that has more than MAX_HIERARCHICAL_CLASSES spanned leaves we
# sub-sample those leaves randomly which is essentially performing class
# selection without taking the hierarchy into account).
MAX_SPANNING_LEAVES_ELIGIBLE = 392


def sample_num_ways_uniformly(num_classes, min_ways, max_ways):
  """Samples a number of ways for an episode uniformly and at random.

  The support of the distribution is [min_ways, num_classes], or
  [min_ways, max_ways] if num_classes > max_ways.

  Args:
    num_classes: int, number of classes.
    min_ways: int, minimum number of ways.
    max_ways: int, maximum number of ways. Only used if num_classes > max_ways.

  Returns:
    num_ways: int, number of ways for the episode.
  """
  max_ways = min(max_ways, num_classes)
  return RNG.randint(low=min_ways, high=max_ways + 1)


def sample_class_ids_uniformly(num_ways, rel_classes):
  """Samples the (relative) class IDs for the episode.

  Args:
    num_ways: int, number of ways for the episode.
    rel_classes: list of int, available class IDs to sample from.

  Returns:
    class_ids: np.array, class IDs for the episode, with values in rel_classes.
  """
  return RNG.choice(rel_classes, num_ways, replace=False)


def compute_num_query(images_per_class, max_num_query, num_support):
  """Computes the number of query examples per class in the episode.

  Query sets are balanced, i.e., contain the same number of examples for each
  class in the episode.

  The number of query examples satisfies the following conditions:
  - it is no greater than `max_num_query`
  - if support size is unspecified, it is at most half the size of the
    smallest class in the episode
  - if support size is specified, it is at most the size of the smallest class
    in the episode minus the max support size.

  Args:
    images_per_class: np.array, number of images for each class.
    max_num_query: int, number of images for each class.
    num_support: int or tuple(int, int), number (or range) of support
      images per class.

  Returns:
    num_query: int, number of query examples per class in the episode.
  """
  if num_support is None:
    if images_per_class.min() < 2:
      raise ValueError('Expected at least 2 images per class.')
    return np.minimum(max_num_query, (images_per_class // 2).min())
  elif isinstance(num_support, int):
    max_support = num_support
  else:
    _, max_support = num_support
  if (images_per_class - max_support).min() < 1:
    raise ValueError(
        'Expected at least {} images per class'.format(max_support + 1))
  return np.minimum(max_num_query, images_per_class.min() - max_support)


def sample_support_set_size(num_remaining_per_class,
                            max_support_size_contrib_per_class,
                            max_support_set_size):
  """Samples the size of the support set in the episode.

  That number is such that:

  * The contribution of each class to the number is no greater than
    `max_support_size_contrib_per_class`.
  * It is no greater than `max_support_set_size`.
  * The support set size is greater than or equal to the number of ways.

  Args:
    num_remaining_per_class: np.array, number of images available for each class
      after taking into account the number of query images.
    max_support_size_contrib_per_class: int, maximum contribution for any given
      class to the support set size. Note that this is not a limit on the number
      of examples of that class in the support set; this is a limit on its
      contribution to computing the support set _size_.
    max_support_set_size: int, maximum size of the support set.

  Returns:
    support_set_size: int, size of the support set in the episode.
  """
  if max_support_set_size < len(num_remaining_per_class):
    raise ValueError('max_support_set_size is too small to have at least one '
                     'support example per class.')
  beta = RNG.uniform()
  support_size_contributions = np.minimum(max_support_size_contrib_per_class,
                                          num_remaining_per_class)
  return np.minimum(
      # Taking the floor and adding one is equivalent to sampling beta uniformly
      # in the (0, 1] interval and taking the ceiling of its product with
      # `support_size_contributions`. This ensures that the support set size is
      # at least as big as the number of ways.
      np.floor(beta * support_size_contributions + 1).sum(),
      max_support_set_size)


def sample_num_support_per_class(images_per_class, num_remaining_per_class,
                                 support_set_size, min_log_weight,
                                 max_log_weight):
  """Samples the number of support examples per class.

  At a high level, we wish the composition to loosely match class frequencies.
  Sampling is done such that:

  * The number of support examples per class is no greater than
    `support_set_size`.
  * The number of support examples per class is no greater than the number of
    remaining examples per class after the query set has been taken into
    account.

  Args:
    images_per_class: np.array, number of images for each class.
    num_remaining_per_class: np.array, number of images available for each class
      after taking into account the number of query images.
    support_set_size: int, size of the support set in the episode.
    min_log_weight: float, minimum log-weight to give to any particular class.
    max_log_weight: float, maximum log-weight to give to any particular class.

  Returns:
    num_support_per_class: np.array, number of support examples for each class.
  """
  if support_set_size < len(num_remaining_per_class):
    raise ValueError('Requesting smaller support set than the number of ways.')
  if np.min(num_remaining_per_class) < 1:
    raise ValueError('Some classes have no remaining examples.')

  # Remaining number of support examples to sample after we guarantee one
  # support example per class.
  remaining_support_set_size = support_set_size - len(num_remaining_per_class)

  unnormalized_proportions = images_per_class * np.exp(
      RNG.uniform(min_log_weight, max_log_weight, size=images_per_class.shape))
  support_set_proportions = (
      unnormalized_proportions / unnormalized_proportions.sum())

  # This guarantees that there is at least one support example per class.
  num_desired_per_class = np.floor(
      support_set_proportions * remaining_support_set_size).astype('int32') + 1

  return np.minimum(num_desired_per_class, num_remaining_per_class)


class EpisodeDescriptionSampler(object):
  """Generates descriptions of Episode composition.

  In particular, for each Episode, it will generate the class IDs (relative to
  the selected split of the dataset) to include, as well as the number of
  support and query examples for each class ID.
  """

  def __init__(self,
               dataset_spec,
               split,
               episode_descr_config,
               pool=None
               ):
    """Initializes an EpisodeDescriptionSampler.episode_config.

    Args:
      dataset_spec: DatasetSpecification, dataset specification.
      split: one of Split.TRAIN, Split.VALID, or Split.TEST.
      episode_descr_config: An instance of EpisodeDescriptionConfig containing
        parameters relating to sampling shots and ways for episodes.
      pool: A string ('train' or 'test') or None, indicating which example-level
        split to select, if the current dataset has them.
     

    Raises:
      RuntimeError: if required parameters are missing.
      ValueError: Inconsistent parameters.
    """
    self.dataset_spec = dataset_spec
    self.split = split
    self.pool = pool
   
    
    self.num_ways = episode_descr_config.num_ways
    self.num_support = episode_descr_config.num_support
    self.num_query = episode_descr_config.num_query
    self.min_ways = episode_descr_config.min_ways
    self.max_ways_upper_bound = episode_descr_config.max_ways_upper_bound
    self.max_num_query = episode_descr_config.max_num_query
    self.max_support_set_size = episode_descr_config.max_support_set_size
    self.max_support_size_contrib_per_class = episode_descr_config.max_support_size_contrib_per_class
    self.min_log_weight = episode_descr_config.min_log_weight
    self.max_log_weight = episode_descr_config.max_log_weight
    self.min_examples_in_class = episode_descr_config.min_examples_in_class

    self.class_set = dataset_spec.get_classes(self.split)
    

    self.num_classes = len(self.class_set)
    # Filter out classes with too few examples
    self._filtered_class_set = []
    # Store (class_id, n_examples) of skipped classes for logging.
    skipped_classes = []
    for class_id in self.class_set:
      n_examples = dataset_spec.get_total_images_per_class(class_id, pool=pool)
      # print(f"Class_id: {class_id}, count: {n_examples}")
      if n_examples < self.min_examples_in_class:
        skipped_classes.append((class_id, n_examples))
      else:
        self._filtered_class_set.append(class_id)
    self.num_filtered_classes = len(self._filtered_class_set)

    if skipped_classes:
      logging.info(
          'Skipping the following classes, which do not have at least '
          '%d examples', self.min_examples_in_class)
    for class_id, n_examples in skipped_classes:
      logging.info('%s (ID=%d, %d examples)',
                   dataset_spec.class_names[class_id], class_id, n_examples)

    if self.min_ways and self.num_filtered_classes < self.min_ways:
      raise ValueError(
          '"min_ways" is set to {}, but split {} of dataset {} only has {} '
          'classes with at least {} examples ({} total), so it is not possible '
          'to create an episode for it. This may have resulted from applying a '
          'restriction on this split of this dataset by specifying '
          'benchmark.restrict_classes or benchmark.min_examples_in_class.'
          .format(self.min_ways, split, dataset_spec.name,
                  self.num_filtered_classes, self.min_examples_in_class,
                  self.num_classes))



  def sample_class_ids(self):
    """Returns the (relative) class IDs for an episode.

    If self.use_dag_hierarchy, it samples them according to a procedure
    informed by the dataset's ontology, otherwise randomly.
    If self.min_examples_in_class > 0, classes with too few examples will not
    be selected.
    """
    prob = [1.0, 0.0]
    
   
    
    if self.num_ways is not None:
      num_ways = self.num_ways
    else:
      num_ways = sample_num_ways_uniformly(
          self.num_filtered_classes,
          min_ways=self.min_ways,
          max_ways=self.max_ways_upper_bound)
    # Filtered class IDs relative to the selected split
    ids_rel = [
        class_id - self.class_set[0] for class_id in self._filtered_class_set
    ]
    episode_classes_rel = sample_class_ids_uniformly(num_ways, ids_rel)

    return episode_classes_rel

  def sample_episode_description(self):
    """Returns the composition of an episode.

    Returns:
      A sequence of `(class_id, num_support, num_query)` tuples, where
        relative `class_id` is an integer in [0, self.num_classes).
    """
    class_ids = self.sample_class_ids()
    images_per_class = np.array([
        self.dataset_spec.get_total_images_per_class(
            self.class_set[cid], pool=self.pool) for cid in class_ids
    ])

    if self.num_query is not None:
      num_query = self.num_query
    else:
      num_query = compute_num_query(
          images_per_class,
          max_num_query=self.max_num_query,
          num_support=self.num_support)

    if self.num_support is not None:
      if isinstance(self.num_support, int):
        if any(self.num_support + num_query > images_per_class):
          raise ValueError('Some classes do not have enough examples.')
        num_support = self.num_support
      else:
        start, end = self.num_support
        if any(end + num_query > images_per_class):
          raise ValueError('The range provided for uniform sampling of the '
                           'number of support examples per class is not valid: '
                           'some classes do not have enough examples.')
        num_support = RNG.randint(low=start, high=end + 1)
      num_support_per_class = [num_support for _ in class_ids]
    else:
      num_remaining_per_class = images_per_class - num_query
      support_set_size = sample_support_set_size(
          num_remaining_per_class,
          self.max_support_size_contrib_per_class,
          max_support_set_size=self.max_support_set_size)
      num_support_per_class = sample_num_support_per_class(
          images_per_class,
          num_remaining_per_class,
          support_set_size,
          min_log_weight=self.min_log_weight,
          max_log_weight=self.max_log_weight)
    # print("Class ids: ")
    # print(class_ids)
    # print("Support size:")
    # print(num_support_per_class)
    return tuple(
        (class_id, num_support, num_query)
        for class_id, num_support in zip(class_ids, num_support_per_class))

  def compute_chunk_sizes(self):
    """Computes the maximal sizes for the flush, support, and query chunks.

    Sequences of dataset IDs are padded with placeholder IDs to make sure they
    can be batched into episodes of equal sizes.

    The "flush" part of the sequence has a size that is upper-bounded by the
    size of the "support" and "query" parts.

    If variable, the size of the "support" part is in the worst case

        max_support_set_size,

    and the size of the "query" part is in the worst case

        max_ways_upper_bound * max_num_query.

    Returns:
      The sizes of the flush, support, and query chunks.
    """
    if self.num_ways is None:
      max_num_ways = self.max_ways_upper_bound
    else:
      max_num_ways = self.num_ways

    if self.num_support is None:
      support_chunk_size = self.max_support_set_size
    elif isinstance(self.num_support, int):
      support_chunk_size = max_num_ways * self.num_support
    else:
      largest_num_support_per_class = self.num_support[1]
      support_chunk_size = max_num_ways * largest_num_support_per_class

    if self.num_query is None:
      max_num_query = self.max_num_query
    else:
      max_num_query = self.num_query
    query_chunk_size = max_num_ways * max_num_query

    flush_chunk_size = support_chunk_size + query_chunk_size
    return (flush_chunk_size, support_chunk_size, query_chunk_size)
