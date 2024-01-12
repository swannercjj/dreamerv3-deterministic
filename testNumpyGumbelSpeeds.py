from dreamerv3 import embodied
from embodied.replay import selectors
from embodied.core import Timer
import numpy as np
# import jax.numpy as jnp
from timeParameterized import TestParameterizedReplay, TestUniformReplay


class NumpyEntirelySelector:
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
    gumbels = self.rng.gumbel(size=len(self.logits))
    idx = np.argmax(self.logits + gumbels)
    return self.keys[idx]

  def __setitem__(self, key, steps):
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      self.logits = np.append(self.logits, self.default_logit_value)

  def __delitem__(self, key):
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index
      if has_waiting:
        other_logits = np.concatenate((self.logits[:index], self.logits[index+1:]))
        new_logit = np.log(np.sum(np.exp(other_logits))) - np.log(len(other_logits))
        self.logits[index] = new_logit
      else:
        last_logit = self.logits[-1]
        self.logits[index] = last_logit
        self.logits = self.logits[:-1]
    elif not has_waiting:
      self.logits = self.logits[:-1]

class ListBackendSelector:
  """A selector that contains a set of parameters(logits) used in item selection.
  """
  def __init__(self, length:int, default_logit_value:float=0, seed=0):
    self.length = length
    self.keys = []
    self.indices = {}
    self.default_logit_value = default_logit_value
    self._logit_list = []
    self.rng = np.random.default_rng(seed)

  @property
  def logits(self):
    return np.array(self._logit_list, dtype=np.single)

  def __call__(self):
    gumbels = self.rng.gumbel(size=len(self._logit_list))
    idx = np.argmax(self.logits + gumbels)
    return self.keys[idx]

  def __setitem__(self, key, steps):
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      self._logit_list.append(self.default_logit_value)

  def __delitem__(self, key):
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index
      if has_waiting:
        other_logits = self._logit_list[:index] + self._logit_list[index+1:]
        new_logit = np.log(np.sum(np.exp(other_logits))) - np.log(len(other_logits))
        self._logit_list[index] = new_logit
      else:
        last_logit = self._logit_list.pop()
        self.logits[index] = last_logit
    elif not has_waiting:
      self._logit_list.pop()

# FASTEST:
class NumpyReduceSelector:
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
    gumbels = self.rng.gumbel(size=len(self.logits))
    idx = np.argmax(self.logits + gumbels)
    return self.keys[idx]

  def __setitem__(self, key, steps):
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      self.logits = np.append(self.logits, self.default_logit_value)

  def __delitem__(self, key):
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index
      if has_waiting:
        other_logits = np.concatenate((self.logits[:index], self.logits[index+1:]))
        new_logit = np.log(np.add.reduce(np.exp(other_logits))) - np.log(len(other_logits))
        self.logits[index] = new_logit
      else:
        last_logit = self.logits[-1]
        self.logits[index] = last_logit
        self.logits = self.logits[:-1]
    elif not has_waiting:
      self.logits = self.logits[:-1]

class NumpyLogAddExpReduceSelector:
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
    gumbels = self.rng.gumbel(size=len(self.logits))
    idx = np.argmax(self.logits + gumbels)
    return self.keys[idx]

  def __setitem__(self, key, steps):
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      self.logits = np.append(self.logits, self.default_logit_value)

  def __delitem__(self, key):
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index
      if has_waiting:
        other_logits = np.concatenate((self.logits[:index], self.logits[index+1:]))
        new_logit = np.logaddexp.reduce(other_logits) - np.log(len(other_logits))
        self.logits[index] = new_logit
      else:
        last_logit = self.logits[-1]
        self.logits[index] = last_logit
        self.logits = self.logits[:-1]
    elif not has_waiting:
      self.logits = self.logits[:-1]

class NumpyReduceDeleteForNewLogitSelector:
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
    gumbels = self.rng.gumbel(size=len(self.logits))
    idx = np.argmax(self.logits + gumbels)
    return self.keys[idx]

  def __setitem__(self, key, steps):
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      self.logits = np.append(self.logits, self.default_logit_value)

  def __delitem__(self, key):
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index
      if has_waiting:
        other_logits = np.delete(self.logits, index)
        new_logit = np.log(np.add.reduce(np.exp(other_logits))) - np.log(len(other_logits))
        self.logits[index] = new_logit
      else:
        last_logit = self.logits[-1]
        self.logits[index] = last_logit
        self.logits = self.logits[:-1]
    elif not has_waiting:
      self.logits = self.logits[:-1]

class NumpyReduceMaskForNewLogitSelector:
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
    gumbels = self.rng.gumbel(size=len(self.logits))
    idx = np.argmax(self.logits + gumbels)
    return self.keys[idx]

  def __setitem__(self, key, steps):
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      self.logits = np.append(self.logits, self.default_logit_value)

  def __delitem__(self, key):
    has_waiting:bool = len(self.keys) > self.length
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index
      if has_waiting:
        logit_len = len(self.logits)
        mask = np.ones(logit_len, dtype=bool)
        mask[index] = False
        new_logit = np.log(np.add.reduce(np.exp(self.logits[mask]))) - np.log(logit_len-1)
        self.logits[index] = new_logit
      else:
        last_logit = self.logits[-1]
        self.logits[index] = last_logit
        self.logits = self.logits[:-1]
    elif not has_waiting:
      self.logits = self.logits[:-1]



class Replay:
  def __init__(self, sampler, length:int, seed=0):
    self.length = length
    self.remover = selectors.Fifo()
    self.sampler = sampler(self.length, seed=seed)
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

# class NumpyEntirelyReplay:
#   def __init__(self, length:int, seed=0):
#     self.length = length
#     self.remover = selectors.Fifo()
#     self.sampler = NumpyEntirelySelector(self.length, seed=seed)
#     self.table = {}
#   def __len__(self) -> int:
#     return len(self.table)
#   def _remove(self, key):
#     del self.table[key]
#     del self.sampler[key]
#     del self.remover[key]
#   def _add(self, item):
#     key = embodied.uuid()
#     self.table[key] = item
#     self.sampler[key] = item
#     self.remover[key] = item
#   def _delete(self):
#     while self.length > 0 and len(self) > self.length:
#       key_to_remove = self.remover()
#       self._remove(key_to_remove)
#   def add(self, item):
#     self._add(item)
#     self._delete()
#   def _sample(self):
#     key = self.sampler()
#     return self.table[key]
#   def dataset(self):
#     while True:
#       yield self._sample()
#   @property
#   def logits(self):
#     return self.sampler.logits

# class ListBackendReplay:
#   def __init__(self, length:int, seed=0):
#     self.length = length
#     self.remover = selectors.Fifo()
#     self.sampler = ListBackendSelector(self.length, seed=seed)
#     self.table = {}
#   def __len__(self) -> int:
#     return len(self.table)
#   def _remove(self, key):
#     del self.table[key]
#     del self.sampler[key]
#     del self.remover[key]
#   def _add(self, item):
#     key = embodied.uuid()
#     self.table[key] = item
#     self.sampler[key] = item
#     self.remover[key] = item
#   def _delete(self):
#     while self.length > 0 and len(self) > self.length:
#       key_to_remove = self.remover()
#       self._remove(key_to_remove)
#   def add(self, item):
#     self._add(item)
#     self._delete()
#   def _sample(self):
#     key = self.sampler()
#     return self.table[key]
#   def dataset(self):
#     while True:
#       yield self._sample()
#   @property
#   def logits(self):
#     return self.sampler.logits

# class NumpyReduceReplay:
#   def __init__(self, length:int, seed=0):
#     self.length = length
#     self.remover = selectors.Fifo()
#     self.sampler = NumpyReduceSelector(self.length, seed=seed)
#     self.table = {}
#   def __len__(self) -> int:
#     return len(self.table)
#   def _remove(self, key):
#     del self.table[key]
#     del self.sampler[key]
#     del self.remover[key]
#   def _add(self, item):
#     key = embodied.uuid()
#     self.table[key] = item
#     self.sampler[key] = item
#     self.remover[key] = item
#   def _delete(self):
#     while self.length > 0 and len(self) > self.length:
#       key_to_remove = self.remover()
#       self._remove(key_to_remove)
#   def add(self, item):
#     self._add(item)
#     self._delete()
#   def _sample(self):
#     key = self.sampler()
#     return self.table[key]
#   def dataset(self):
#     while True:
#       yield self._sample()
#   @property
#   def logits(self):
#     return self.sampler.logits



def timeit(replay, timer, experience_len, n_experience, n_samples_per_step, n_samples_after):
  experience = [str(i) for i in range(experience_len)] * n_experience
  timer.wrap('replay', replay, ['_add', '_sample', '_delete'])
  for i,exp in enumerate(experience):
    replay.add(exp)
    for _ in range(n_samples_per_step):
      s = replay._sample()
    # if i % (len(experience) // 10) == 0:
    #   print(f'done step {i}')
  for _ in range(n_samples_after):
    s = replay._sample()
  return timer

def printStats(name, timer):
  print('================================')
  print(f"{name}:")
  timer.stats(log=True)

def trainReplay(name, replay):
  EXP_LEN = 300
  NUM_TIMES_EXP = 10
  N_SAMPLES_PER_STEP = 10
  N_SAMPLES_AFTER = 0
  timer = timeit(replay, Timer(), EXP_LEN, NUM_TIMES_EXP, N_SAMPLES_PER_STEP, N_SAMPLES_AFTER)
  printStats(name, timer)


if __name__ == '__main__':
  CAPACITY = 250
  SEED = 0
  trainReplay("UNIFORM JAX", TestUniformReplay(CAPACITY, seed=SEED))
  # trainReplay("PARAMETERIZED JAX", TestParameterizedReplay(CAPACITY, seed=SEED))
  trainReplay("NUMPY ENTIRELY", Replay(NumpyEntirelySelector, CAPACITY, seed=SEED))
  # trainReplay("LIST BACKEND", Replay(ListBackendSelector, CAPACITY, seed=SEED))
  trainReplay("NUMPY WITH REDUCE", Replay(NumpyReduceSelector, CAPACITY, seed=SEED))
  # trainReplay("NUMPY WITH LOGADDEXP REDUCE", Replay(NumpyLogAddExpReduceSelector, CAPACITY, seed=SEED))
  trainReplay("NUMPY WITH REDUCE AND DELETE FOR NEW LOGIT", Replay(NumpyReduceDeleteForNewLogitSelector, CAPACITY, seed=SEED))
  trainReplay("NUMPY WITH REDUCE AND MASK FOR NEW LOGIT", Replay(NumpyReduceMaskForNewLogitSelector, CAPACITY, seed=SEED))