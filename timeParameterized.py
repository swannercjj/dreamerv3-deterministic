from dreamerv3 import embodied
from embodied.replay import selectors
from embodied.core import Timer
import numpy as np
# import matplotlib.pyplot as plt
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
  def _add(self, item):
    key = embodied.uuid()
    self.table[key] = item
    self.sampler[key] = item
    self.remover[key] = item
  def _delete(self):
    while self.length > 0 and len(self) > self.length:
      key_to_remove = self.remover()
      self._remove(key_to_remove)
  def add(self, item):
    self._add(item)
    self._delete()
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


class TestUniformReplay:
  def __init__(self, length:int, seed=0):
    self.length = length
    self.remover = selectors.Fifo()
    self.sampler = selectors.Uniform(seed=seed)
    self.table = {}
  def __len__(self) -> int:
    return len(self.table)
  def _remove(self, key):
    del self.table[key]
    del self.sampler[key]
    del self.remover[key]
  def _add(self, item):
    key = embodied.uuid()
    self.table[key] = item
    self.sampler[key] = item
    self.remover[key] = item
  def _delete(self):
    while self.length > 0 and len(self) > self.length:
      key_to_remove = self.remover()
      self._remove(key_to_remove)
  def add(self, item):
    self._add(item)
    self._delete()
  def _sample(self):
    key = self.sampler()
    return self.table[key]
  def dataset(self):
    while True:
      yield self._sample()


def timeit(replay, timer, experience_len, n_experience, n_samples_per_step, n_samples_after):
  experience = [str(i) for i in range(experience_len)] * n_experience
  timer.wrap('replay', replay, ['_add', '_sample', '_delete'])
  for i,exp in enumerate(experience):
    replay.add(exp)
    for _ in range(n_samples_per_step):
      s = replay._sample()
    if i % (len(experience) // 10) == 0:
      print(f'done step {i}')
  for _ in range(n_samples_after):
    s = replay._sample()
  return timer


if __name__ == '__main__':
  CAPACITY = 250
  SEED = 0
  EXP_LEN = 300
  NUM_TIMES_EXP = 2
  N_SAMPLES_PER_STEP = 10
  N_SAMPLES_AFTER = 0
  uniformTimer = timeit(TestUniformReplay(CAPACITY, seed=SEED), Timer(), EXP_LEN, NUM_TIMES_EXP, N_SAMPLES_PER_STEP, N_SAMPLES_AFTER)
  print('DONE TRAINING UNIFORM\n')
  parameterizedTimer = timeit(TestParameterizedReplay(CAPACITY, seed=SEED), Timer(), EXP_LEN, NUM_TIMES_EXP, N_SAMPLES_PER_STEP, N_SAMPLES_AFTER)
  print('DONE TRAINING PARAMETERIZED\n')

  uniformTimer.stats(log=True)
  print('================================')
  parameterizedTimer.stats(log=True)