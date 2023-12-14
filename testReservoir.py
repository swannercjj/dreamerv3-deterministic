from dreamerv3 import embodied
from embodied.replay import selectors
import numpy as np
import matplotlib.pyplot as plt


class TestReservoirReplay:
  def __init__(self, length:int, seed=0):
    self.length = length
    self.remover = selectors.Reservoir(self.length, seed=seed)
    self.sampler = selectors.Uniform(seed=seed)
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


def testBufferCorrectness():
  # test the replay buffer
  buffer_length = 100
  n_trials = 1000
  experience = [str(i) for i in range(0,200)]
  in_buffer_at_step = [
    [
      {
        exp:False for exp in experience
      }
      for step in range(len(experience))
    ]
    for trial in range(n_trials)
  ]
  
  for trial in range(n_trials):
    replay = TestReservoirReplay(buffer_length, seed=trial)
    for i, exp in enumerate(experience):
      # Measure the number of times an input is in the buffer before the (i+1)th input is processed
      for key,item in replay.table.items():
        in_buffer_at_step[trial][i][item] = True # It is in the buffer before the element `i` is added
      replay.add(exp) # Add element `i`, which is the (i+1)th input
  
  # Calculate the probabilities that an element is in the buffer before the (i+1)th input is processed
  probabilities = [
    dict()
    for step in range(len(experience))
  ]
  for step in range(len(experience)):
    for exp in experience:
      num = 0; den = 0
      for trial in range(n_trials):
        if exp in in_buffer_at_step[trial][step]:
          if in_buffer_at_step[trial][step][exp] == True:
            num += 1
          den += 1
      probabilities[step][exp] = num / den if den != 0 else 0
  
  # Plot the results
  steps_to_plot = [50, 102, 150, 199]
  for pltindex, step in enumerate(steps_to_plot):
    plt.subplot(len(steps_to_plot), 1, pltindex + 1)
    x = experience
    y = [probabilities[step].get(exp, 0) for exp in x]
    plt.bar(x, height=y)
    plt.ylim(0.0, 1.0)
    k_over_i = buffer_length/step
    plt.axhline(y=k_over_i, color='r', linestyle='--') # k/i
    non_zero_y = [v for v in y if v > 0]
    average_prob = sum(non_zero_y) / len(non_zero_y)
    pdiff = lambda v1,v2: abs(v1 - v2) / ((v1 + v2) / 2) * 100
    k_over_i_plus_one = buffer_length / (step + 1)
    print(f"STEP {step}:   k/i = {k_over_i}, average_prob = {average_prob}, diff = {pdiff(k_over_i,average_prob):.5f},   k/(i+1) = {k_over_i_plus_one}, diff = {pdiff(k_over_i_plus_one, average_prob):.5f}")
  plt.tight_layout()
  plt.savefig('fig.png')

if __name__ == '__main__':
  testBufferCorrectness()
