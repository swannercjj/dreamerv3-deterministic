from dreamerv3 import embodied
from embodied.replay import selectors
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


class TestParameterizedReplay:
  def __init__(self, length:int, seed=0):
    self.length = length
    self.remover = selectors.Fifo()
    self.sampler = selectors.Parameterized(self.length, seed=seed)
    self.table = {}
  def __len__(self) -> int:
    return len(self.table)
  def _remove(self, key):
    del self.table[key]
    del self.sampler[key]
    del self.remover[key]
  def add(self, item):
    key = embodied.uuid()
    self.table[key] = item
    self.sampler[key] = item
    self.remover[key] = item
    while self.length > 0 and len(self) > self.length:
      key_to_remove = self.remover()
      self._remove(key_to_remove)
  def _sample(self):
    key = self.sampler()
    return self.table[key]
  def dataset(self):
    while True:
      yield self._sample()
  @property
  def logits(self):
    return self.sampler.logits
  @logits.setter
  def logits(self, value):
    self.sampler.logits = value


def testBufferLogitSetting():
  s = selectors.Parameterized(5, 0)
  for i in range(1,6):
    s[i] = 0
    s.logit_dict[i] = i
  print(s.keys, s.logits, s.logit_dict)
  s[6] = 0
  print(s.keys, s.logits, s.logit_dict)
  del s[2]
  print(s.keys, s.logits, s.logit_dict)
  s.logits = s.logits * 2
  print(s.keys, s.logits, s.logit_dict)

def testBufferUniformLogits():
  buffer_length = 20
  n_samples = 3000
  experience = [str(i) for i in range(0,25)]
  n_times_sampled = {
    exp: 0 for exp in experience
  }
  # Add the experience
  t = TestParameterizedReplay(buffer_length, seed=0)
  for exp in experience:
    t.add(exp)
  # Sample
  for sample_number in range(n_samples):
    s = t._sample()
    n_times_sampled[s] += 1
  # Plot the results
  x = experience
  y = [n_times_sampled[exp] for exp in x]
  plt.bar(x, height=y)
  plt.ylim(0, n_samples)
  k_over_i = n_samples / buffer_length
  plt.axhline(y=k_over_i, color='r', linestyle='--') # k/i
  plt.tight_layout()
  plt.savefig('testFigures/fig_parameterized_uniform_logits.png')
  print('DONE')

def testBufferEvenBiased():
  buffer_length = 20
  n_samples = 3000
  experience = [str(i) for i in range(0,25)]
  n_times_sampled = {
    exp: 0 for exp in experience
  }
  # Add the experience
  t = TestParameterizedReplay(buffer_length, seed=0)
  for exp in experience:
    t.add(exp)
  # Bias the logits towards even-indexed experience
  t.logits = t.logits + jnp.array([2 * (1 - (i % 2)) for i in range(buffer_length)], dtype=float)
  # Sample
  for sample_number in range(n_samples):
    s = t._sample()
    n_times_sampled[s] += 1
  # Plot the results
  x = experience
  y = [n_times_sampled[exp] for exp in x]
  plt.bar(x, height=y)
  plt.ylim(0, n_samples)
  k_over_i = n_samples / buffer_length
  plt.axhline(y=k_over_i, color='r', linestyle='--') # k/i
  plt.tight_layout()
  plt.savefig('testFigures/fig_parameterized_even_biased_logits.png')
  print('DONE')


if __name__ == '__main__':
  # testBufferLogitSetting()
  # testBufferUniformLogits()
  testBufferEvenBiased()
