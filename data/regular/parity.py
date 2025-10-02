
#
# Copyright 2025 - IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

"""Compute whether the number of 1s in a string is even."""

import torch as t
import torch.nn.functional as F
from data import task

class ParityCheck(task.GeneralizationTask):
  def __init__(self, batch_size: int, length: int, randomize: bool, device: str):
    self.batch_size = batch_size
    self.length = length
    self.randomize = randomize
    self.device = device

  def sample_batch(self, split, batch_size: int = None, length: int = None) -> task.Batch:
    """Returns a batch of strings and the expected class."""
    """
    This generate method is not good. The number of 1s in the string should be uniformly
    """
    batch_size = batch_size or self.batch_size
    length = length or self.length
    if self.randomize:
      length = t.randint(low=1, high=self.length + 1, size=()).item()
    strings = t.randint(low=0, high=2, size=(batch_size, length))
    n_b = t.sum(strings, axis=1) % 2
    X = strings.to(self.device)
    Y = -t.ones((batch_size, length), dtype=t.long, device=self.device)
    Y[:, -1] = n_b
    return X, Y

  @property
  def input_size(self) -> int:
    """Returns the input size for the models."""
    return 2

  @property
  def output_size(self) -> int:
    """Returns the output size for the models."""
    return 2
  
  @property
  def vocabulary_size(self) -> int:
    """Returns the vocabulary size for the models."""
    return 2
