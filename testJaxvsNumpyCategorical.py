from dreamerv3 import embodied
from embodied.core import Timer

import numpy as np
import jax
import jax.numpy as jnp


class TestJax:
  def __init__(self, logits:list, seed=0):
    self._logit_list = logits.copy()
    self.rng = jax.random.PRNGKey(seed)

  def get_logits(self):
    return jnp.array(self._logit_list, dtype=float)

  def get_sample(self, logits):
    idx = int(jax.random.categorical(self.rng, logits))
    self.rng, _ = jax.random.split(self.rng)
    return idx

  def sample(self):
    logits = self.get_logits()
    idx = self.get_sample(logits)
    return idx

class TestNumpy:
  def __init__(self, logits:list, seed=0):
    self.logits = np.array(logits, dtype=np.single)
    self.rng = np.random.default_rng(seed=seed)
  
  def sample(self):
    gumbels = self.rng.gumbel(size=len(self.logits))
    idx = np.argmax(self.logits + gumbels)
    return idx


if __name__ == '__main__':
  logits = [0] * 250
  seed = 0
  n_samples = 100000
  arbitrary_number = 0
  j, n = TestJax(logits, seed), TestNumpy(logits, seed)
  timer = Timer()
  timer.wrap('JAX', j, ['sample', 'get_logits', 'get_sample'])
  timer.wrap('numpy', n, ['sample'])

  for sample_num in range(n_samples):
    js, ns = j.sample(), n.sample()
    if ns > js:
      arbitrary_number += 1
    if sample_num % (n_samples // 15) == 0:
      print(f'done step {sample_num}/{n_samples} ({(sample_num/n_samples*100):>5.2f}%)')
  
  print(arbitrary_number)
  timer.stats(log=True)
