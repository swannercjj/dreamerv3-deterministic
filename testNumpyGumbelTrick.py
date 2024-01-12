import numpy as np
import matplotlib.pyplot as plt


class TestNumpy:
  def __init__(self, logits:list, seed=0):
    self.logits = np.array(logits, dtype=np.single)
    self.rng = np.random.default_rng(seed=seed)
  
  def sample(self):
    gumbels = self.rng.gumbel(size=len(self.logits))
    idx = np.argmax(self.logits + gumbels)
    return idx


def runTest(test_name, replay, n_samples, num_prints):
  counts = [0 for idx in range(len(replay.logits))]
  for sample_num in range(n_samples):
    idx = replay.sample()
    counts[idx] += 1
    if sample_num % (n_samples // num_prints) == 0:
      print(f'{test_name}: done step {sample_num}/{n_samples} ({(sample_num/n_samples*100):>5.2f}%)')
  print(f'{test_name} done')
  return counts

def testUniform(num_logits:int, n_samples:int, seed=0):
  logits = [5] * num_logits
  replay = TestNumpy(logits, seed=seed)
  counts = runTest('UNIFORM', replay, n_samples, 15)
  plt.bar(list(range(num_logits)), height=counts)
  plt.ylim(0, max(counts) * 1.2)
  plt.axhline(y=n_samples/num_logits, color='r', linestyle='--') # k/i
  plt.tight_layout()
  plt.savefig('testFigures/gumbelTestUniformLogits.png')

def testEvenBiased(num_logits:int, n_samples:int, seed=0):
  logits = [2 if i % 2 == 0 else 1 for i in range(num_logits)]
  replay = TestNumpy(logits, seed=seed)
  counts = runTest('EVEN BIASED', replay, n_samples, 15)
  plt.bar(list(range(num_logits)), height=counts)
  plt.ylim(0, max(counts) * 1.2)
  bottom_sum = sum(np.exp(logits))
  expected = n_samples * np.array(np.exp(logits) / bottom_sum, dtype=float)
  plt.plot(list(range(num_logits)), expected, 'r--')
  plt.tight_layout()
  plt.savefig('testFigures/gumbelTestEvenBiasedLogits.png')

def testDecreasingLogits(num_logits:int, n_samples:int, seed=0):
  logits = [0.995**i for i in range(num_logits)]
  replay = TestNumpy(logits, seed=seed)
  counts = runTest('DECREASING', replay, n_samples, 15)
  plt.bar(list(range(num_logits)), height=counts)
  plt.ylim(0, max(counts) * 1.2)
  bottom_sum = sum(np.exp(logits))
  expected = n_samples * np.array(np.exp(logits) / bottom_sum, dtype=float)
  plt.plot(list(range(num_logits)), expected, 'r--')
  plt.tight_layout()
  plt.savefig('testFigures/gumbelTestDecreasingLogits.png')


if __name__ == '__main__':
  # testUniform(100, 1000000, 0)
  # testEvenBiased(100, 1000000, 0)
  testDecreasingLogits(100, 1000000, 0)