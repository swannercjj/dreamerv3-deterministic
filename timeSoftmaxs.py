from dreamerv3 import embodied
from embodied.replay import selectors
from embodied.core import Timer
import numpy as np
# import matplotlib.pyplot as plt
import jax.numpy as jnp


class Softmax:
  def __init__(self):
    pass

  def eDotSum(self, logits): # roughly equal
    e = np.exp(logits)
    return e / e.sum()

  def npDotSum(self, logits): # roughly equal
    e = np.exp(logits)
    return e / np.sum(e)
  
  def asWeGo(self, logits): # slowest
    length = len(logits)
    e = np.zeros(length, dtype=np.single)
    s = 0
    for i,l in enumerate(logits):
      v = np.exp(l)
      e[i] = v
      s += v
    return e / s

def timeit(timer, n_calls, n_logits):
  softmax = Softmax()
  timer.wrap('softmax', softmax, ['eDotSum', 'npDotSum', 'asWeGo'])

  for i in range(n_calls):
    logits = np.random.random(size=n_logits)
    probs1 = softmax.eDotSum(logits)
    probs2 = softmax.npDotSum(logits)
    probs3 = softmax.asWeGo(logits)
  return timer


if __name__ == '__main__':
  NUM_CALLS_PER_LOGIT_LEN = 100
  NUM_LOGITS = [1000,10000,100000]#,1000000]

  for num_logits in NUM_LOGITS:
    timer = timeit(Timer(), NUM_CALLS_PER_LOGIT_LEN, num_logits)
    print('=================================================================')
    print(f'NUM_LOGITS={num_logits} with NUM_TRIALS={NUM_CALLS_PER_LOGIT_LEN}')
    timer.stats(log=True)
    print('')