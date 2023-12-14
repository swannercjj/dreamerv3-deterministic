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
    if index != len(self.keys): # This is to move the last element to replace the element that was just removed
      self.keys[index] = last
      self.indices[last] = index


class Parameterized:
  """A selector that contains a set of parameters(logits) used in item selection.
  """
  def __init__(self, length:int, default_logit_value:float=0, seed=0):
    # jax array of parameters
    # keep track of index from jax array to keys array
    self.length = length
    self.keys = []
    self.indices = {}
    self.default_logit_value = default_logit_value
    # Should this be increasing in length with `keys` or does it remain the same length?
    self.logit_dict = {} # Assign a dictionary from key -> logit regardless of order
    self.rng = jax.random.PRNGKey(seed)

  @property
  def logits(self):
    return jnp.array([self.logit_dict[k] for k in self.keys[:self.length]], dtype=float)

  @logits.setter
  def logits(self, value):
    if len(value) > len(self.keys) or len(value) > self.length:
      raise ValueError("Cannot set logits to a longer length than keys or self.length.")
    self.logit_dict.clear()
    for key,logit in zip(self.keys[:self.length], value):
      self.logit_dict[key] = logit

  def __call__(self):
    # jax.random.categorical
    idx = int(jax.random.categorical(self.rng, self.logits))
    self.rng, _ = jax.random.split(self.rng) # split the PRNG key
    return self.keys[idx]

  def __setitem__(self, key, steps):
    # add something to the back of the array of keys
    self.indices[key] = len(self.keys)
    self.keys.append(key)
    if len(self.keys) <= self.length:
      # we want the first {length} items added to have the default logit value
      self.logit_dict[key] = self.default_logit_value

  def __delitem__(self, key):
    # assign the logit to something waiting to be inside the parlogitameterized keys
    index = self.indices.pop(key)
    last = self.keys.pop()
    if key in self.logit_dict:
      del self.logit_dict[key] # Get rid of the logit value for the thing we're deleting
    if index != len(self.keys): # If the element we're removing isn't the last one ...
      # Then there's a swap that will occur
      # Move the last element into the spot we are removing
      self.keys[index] = last
      self.indices[last] = index
      if last not in self.logit_dict:
        logits = jnp.array([self.logit_dict[k] for idx,k in enumerate(self.keys[:self.length]) if idx != index], dtype=float)
        # Assign a new logit to the element we previously didn't have a logit for
        new_logit = jax.scipy.special.logsumexp(logits) - jnp.log(len(logits))
        self.logit_dict[last] = new_logit
