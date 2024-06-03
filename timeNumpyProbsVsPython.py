from dreamerv3 import embodied
from embodied.replay import selectors
from embodied.core import Timer
import numpy as np
# import matplotlib.pyplot as plt
import jax.numpy as jnp
from time import time
import random


class ParameterizedNumpySoftmax:
  """A selector that contains a set of parameters(logits) used in item selection.
  """
  def __init__(self, length:int, default_logit_value:float=0, seed=0):
    self.length = length
    self.keys = []
    self.indices = {}
    self.default_logit_value = default_logit_value
    self.logits = np.array([], dtype=np.single)
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    # https://stackoverflow.com/questions/58339083/how-to-sample-from-a-log-probability-distribution
    e = np.exp(self.logits)
    idx = self.rng.choice(len(self.logits), p=e/np.sum(e))
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

class ParameterizedNumpyProbabilities:
  """A selector that contains a set of parameters(logits) used in item selection.
  """
  def __init__(self, length:int, default_logit_value:float=0, seed=0):
    self.length = length
    self.keys = []
    self.indices = {}
    self.default_logit_value = default_logit_value
    self.probs = np.array([], dtype=np.single)
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    # https://stackoverflow.com/questions/58339083/how-to-sample-from-a-log-probability-distribution
    idx = self.rng.choice(len(self.probs), p=self.probs) # assumes probs are normalized
    return self.keys[idx]

  def __setitem__(self, key, steps):
    # add something to the back of the array of keys
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      # we want the first {length} items added to have a default probability value
      new_prob = 1 if len(self.probs) == 0 else 1 / len(self.probs)
      self.probs = np.append(self.probs, new_prob)
      self.probs = self.probs / np.sum(self.probs) # normalize

  def __delitem__(self, key):
    # TODO -- does this ever get called before the buffer is full? Does that matteer?
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys): # If the element is something in the middle of the buffer
      # Move the last element into the spot we are removing
      self.keys[index] = last
      self.indices[last] = index # update the index
      if has_waiting:
        # The element we're moving does not have a probability yet
        # We need to calculate a value for the new probability
        self.probs[index] = 1 / self.length
        self.probs = self.probs / np.sum(self.probs) # normalize
      else:
        # The element we're moving already has a probability assigned; move it
        self.probs[index] = self.probs[-1]
        self.probs = self.probs[:-1]
        self.probs = self.probs / np.sum(self.probs) # normalize
    elif not has_waiting: # The item we're deleting is the last one
      # And there's no element waiting to have a probability assigned
      # So the thing we're deleting has a probability already, so we delete it
      self.probs = self.probs[:-1]
      self.probs = self.probs / np.sum(self.probs) # normalize

class ParameterizedPythonProbabilities:
  """A selector that contains a set of parameters(logits) used in item selection.
  """
  def __init__(self, length:int, default_logit_value:float=0, seed=0):
    self.length = length
    self.keys = []
    self.indices = {}
    self.probs = np.array([], dtype=np.single)
    self.rng = random.Random(seed)

  def __call__(self):
    idx = self.rng.choices(range(len(self.probs)), weights=self.probs)[0]
    return self.keys[idx]

  def __setitem__(self, key, steps):
    # add something to the back of the array of keys
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      # we want the first {length} items added to have a default probability value
      new_prob = 1 if len(self.probs) == 0 else 1 / len(self.probs)
      self.probs = np.append(self.probs, new_prob)
      self.probs = self.probs / np.sum(self.probs) # normalize

  def __delitem__(self, key):
    # TODO -- does this ever get called before the buffer is full? Does that matteer?
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys): # If the element is something in the middle of the buffer
      # Move the last element into the spot we are removing
      self.keys[index] = last
      self.indices[last] = index # update the index
      if has_waiting:
        # The element we're moving does not have a probability yet
        # We need to calculate a value for the new probability
        self.probs[index] = 1 / self.length
        self.probs = self.probs / np.sum(self.probs) # normalize
      else:
        # The element we're moving already has a probability assigned; move it
        self.probs[index] = self.probs[-1]
        self.probs = self.probs[:-1]
        self.probs = self.probs / np.sum(self.probs) # normalize
    elif not has_waiting: # The item we're deleting is the last one
      # And there's no element waiting to have a probability assigned
      # So the thing we're deleting has a probability already, so we delete it
      self.probs = self.probs[:-1]
      self.probs = self.probs / np.sum(self.probs) # normalize



class TestSoftmaxReplay:
  def __init__(self, length:int, seed=0):
    self.length = length
    self.remover = selectors.Fifo()
    self.sampler = ParameterizedNumpySoftmax(self.length, seed=seed)
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

class TestNumpyProbabilitiesReplay:
  def __init__(self, length:int, seed=0):
    self.length = length
    self.remover = selectors.Fifo()
    self.sampler = ParameterizedNumpyProbabilities(self.length, seed=seed)
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

class TestPythonProbabilitiesReplay:
  def __init__(self, length:int, seed=0):
    self.length = length
    self.remover = selectors.Fifo()
    self.sampler = ParameterizedPythonProbabilities(self.length, seed=seed)
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
  timer.wrap('replay', replay, ['_add', '_sample', '_delete'])
  start_time = time()
  for i,exp in enumerate(experience):
    replay.add(exp)
    for _ in range(n_samples_per_step):
      s = replay._sample()
    if i % (len(experience) // 10) == 0:
      print(f'done step {i}')
  for _ in range(n_samples_after):
    s = replay._sample()
  end_time = time()
  return timer, end_time - start_time

def printResults(results):
  longest_name_len = max([len(name) for name,_,_ in results])
  for name,timer,total_time in results:
    print(f'================{name:^{longest_name_len}s}================')
    print(f'Total Time: {total_time:.5f} seconds')
    timer.stats(log=True)

def trainReplays(replays):
  CAPACITY = 100000
  SEED = 0
  EXP_LEN = 500000
  NUM_TIMES_EXP = 1
  N_SAMPLES_PER_STEP = 1
  N_SAMPLES_AFTER = 0
  timing = []
  for name, replay in replays:
    timer, total_time = timeit(replay(CAPACITY, seed=SEED), Timer(), EXP_LEN, NUM_TIMES_EXP, N_SAMPLES_PER_STEP, N_SAMPLES_AFTER)
    timing.append((name,timer,total_time))
    print(f'DONE TRAINING {name}\n')
  printResults(timing)
    

if __name__ == '__main__':
  trainReplays([
    ('SOFTMAX', TestSoftmaxReplay),
    ('NUMPY PROBABILITIES', TestNumpyProbabilitiesReplay),
    ('PYTHON PROBABILITIES', TestPythonProbabilitiesReplay)
  ])