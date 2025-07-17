from . import generic
from . import selectors
from . import limiters

from collections import defaultdict
import numpy as np
import jax.numpy as jnp

class Uniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0):
    if samples_per_insert:
      limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      limiter = limiters.MinSize(min_size)
    assert not capacity or min_size <= capacity
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiter,
        directory=directory,
        online=online,
        chunks=chunks,
    )


class ParameterizedFifo(generic.Generic):
  def __init__(
      self, length, capacity=None, directory=None, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0):
    if samples_per_insert:
      limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      limiter = limiters.MinSize(min_size)
    assert not capacity or min_size <= capacity
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Parameterized(capacity, seed),
        limiter=limiter,
        directory=directory,
        online=online,
        chunks=chunks,
    )
  def sequences_to_lists_of_keys(self):
    # converts the table of keys->sequences to a table of sequence->list[key]
    # so the table is now unique and you can easily access all the keys associated with a sequence
    ret = defaultdict(lambda: [])
    for key,seq in self.table.items():
      ret[seq].append(key)
    return ret


class ReservoirReplay: # \mathcal{B}
  def __init__(self, capacity:int, seed):
    self.probs = np.array([], dtype=np.single)
    self.table = []
    self.capacity = capacity # k
    self.i = 0
    self.rng = np.random.default_rng(seed)
  def add(self, state):
    if self.i >= self.capacity:
      j = self.rng.randint(0, self.i) # [0,i)
      if j < self.capacity:
        # replace = remove + add
        hashable_state = frozenset({k:tuple(np.array(v)) for k,v in state.items()})
        self.table[j] = hashable_state
        self.probs[j] = 1 / self.capacity
        self.probs = self.probs / np.sum(self.probs) # normalize
    else:
      # add to fill buffer
      hashable_state = frozenset({k:tuple(np.array(v)) for k,v in state.items()})
      self.table.append(hashable_state)
      # we want the first {length} items added to have a default probability value
      if self.i == 0:
        new_prob = 1
      else:
        new_prob = 1 / len(self.table)
      self.probs = np.append(self.probs, new_prob)
      self.probs = self.probs / np.sum(self.probs) # normalize
    self.i += 1
  def _single_sample(self):
    idx = self.rng.choice(len(self.probs), p=self.probs) # assumes probs are normalized
    hashed_state = self.keys[idx]
    state = {k:jnp.array(v) for k,v in dict(hashed_state).items()}
    return state
  def sample(self, batch_size:int): # should be config.batch_steps
    states = [self._single_sample() for _ in range(batch_size)]
    batch = {k: jnp.stack([state[k] for state in states], 0) for k in states[0]} # TODO -- maybe this can be in the Batcher
    return batch
  def as_unique(self):
    r = ReservoirReplay(self.capacity, self.seed)
    unique_elements = set()
    summed_probs = defaultdict(lambda: 0.0)
    for s,p in zip(self.table, self.probs):
      summed_probs[s] += p
    r.table = list(unique_elements)
    r.probs = np.array([summed_probs[s] for s in r.table], dtype=np.single)
    return r
