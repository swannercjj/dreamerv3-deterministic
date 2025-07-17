from collections import deque

import numpy as np
import jax
import jax.numpy as jnp

################################################################
# REMOVERS
################################################################

class Fifo:

  def __init__(self):
    self.queue = deque()

  def __call__(self):
    return self.queue[0]

  def __setitem__(self, key, steps):
    self.queue.append(key)

  def __delitem__(self, key):
    if self.queue[0] == key:
      self.queue.popleft()
    else:
      # TODO: This branch is unused but very slow.
      self.queue.remove(key)


class Reservoir:
  """A reservoir selector for removal using Algorithm R
  https://en.wikipedia.org/wiki/Reservoir_sampling
  """
  def __init__(self, k:int, seed=0):
    self.k = k # The fixed length of the buffer, >= 1
    self.keys = []
    self.rng = np.random.default_rng(seed)
    self.i = 0
  
  def __call__(self):
    return self.keys[-1]

  def __setitem__(self, key, steps):
    if self.i < self.k: # Fill up the reservoir
      self.keys.append(key)
      self.i += 1
      return
    self.keys.append(key)
    j = int((self.i + 1) * self.rng.random()) # generate a random integer from [0,i]
    if j < self.k:
      # Swap items j and the new key
      self.keys[j], self.keys[-1] = self.keys[-1], self.keys[j]
    self.i += 1

  def __delitem__(self, key):
    # Remove the key from self.keys
    if key == self.keys[-1]:
      del self.keys[-1]
    else:
      # Should never occur
      self.keys.remove(key)


################################################################
# SELECTORS
################################################################

class Uniform:

  def __init__(self, seed=0):
    self.indices = {}
    self.keys = []
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    index = self.rng.integers(0, len(self.keys)).item()
    return self.keys[index]

  def __setitem__(self, key, steps):
    self.indices[key] = len(self.keys)
    self.keys.append(key)

  def __delitem__(self, key):
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index


class Parameterized:
  """A selector that contains a set of parameters(logits) used in item selection.
  """
  def __init__(self, length:int, seed=0):
    self.length = length
    self.keys = []
    self.indices = {}
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
  
  def get_prob(self, key):
    return self.probs[self.indices[key]]