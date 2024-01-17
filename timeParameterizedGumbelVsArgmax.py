from dreamerv3 import embodied
from embodied.replay import selectors
from embodied.core import Timer
import numpy as np
# import matplotlib.pyplot as plt
import jax.numpy as jnp


class Parameterized:
  """A selector that contains a set of parameters(logits) used in item selection.
  """
  def __init__(self, length:int, default_logit_value:float=0, seed=0):
    self.length = length
    self.keys = []
    self.indices = {}
    self.default_logit_value = default_logit_value
    self.logits = np.array([], dtype=np.single)
    self.rng = np.random.default_rng(seed)

  def makeGumbels(self):
    return self.rng.gumbel(size=len(self.logits))

  def takeArgmax(self, gumbels):
    return np.argmax(self.logits + gumbels)

  def __call__(self):
    # https://stackoverflow.com/questions/58339083/how-to-sample-from-a-log-probability-distribution
    gumbels = self.makeGumbels()
    idx = self.takeArgmax(gumbels)
    return self.keys[idx]

  def __setitem__(self, key, steps):
    # add something to the back of the array of keys
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      # we want the first {length} items added to have the default logit value
      self.logits = np.append(self.logits, self.default_logit_value)

  def __delitem__(self, key):
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys): # If the element is something in the middle of the buffer
      # Move the last element into the spot we are removing
      self.keys[index] = last
      self.indices[last] = index # update the index
      if has_waiting:
        # The element we're moving does not have a logit yet
        # We need to calculate a value for the new logit
        other_logits = np.concatenate((self.logits[:index], self.logits[index+1:]))
        # using np.log(np.add.reduce(np.exp(logits))) is faster than np.logaddexp.reduce(logits)
        new_logit = np.log(np.add.reduce(np.exp(other_logits))) - np.log(len(other_logits))
        self.logits[index] = new_logit
      else:
        # The element we're moving already has a logit assigned; move it
        self.logits[index] = self.logits[-1]
        self.logits = self.logits[:-1]
    elif not has_waiting: # The item we're deleting is the last one
      # And there's no element waiting to have a logit assigned
      # So the thing we're deleting has a logit already, so we delete it
      self.logits = self.logits[:-1]


class TestParameterizedReplay:
  def __init__(self, length:int, seed=0):
    self.length = length
    self.remover = selectors.Fifo()
    self.sampler = Parameterized(self.length, seed=seed)
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


def timeit(replay, timer, experience_len, n_experience, n_samples_per_step, n_samples_after):
  experience = [str(i) for i in range(experience_len)] * n_experience
  timer.wrap('replay.sampler', replay.sampler, ['makeGumbels', 'takeArgmax'])
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
  CAPACITY = 50000
  SEED = 0
  EXP_LEN = 250000
  NUM_TIMES_EXP = 1
  N_SAMPLES_PER_STEP = 1
  N_SAMPLES_AFTER = 0
  parameterizedTimer = timeit(TestParameterizedReplay(CAPACITY, seed=SEED), Timer(), EXP_LEN, NUM_TIMES_EXP, N_SAMPLES_PER_STEP, N_SAMPLES_AFTER)
  print('DONE TRAINING\n')

  print('================================')
  parameterizedTimer.stats(log=True)