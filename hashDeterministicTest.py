import pickle
import sys

# Last session:
# 0       0       -8458139203682520985    753925839

is_writing = False

if is_writing:
  idx = int(sys.argv[1]) # this may make it all deterministic
  _seed = 10
  _episode = 100

  seed_before_mod = hash((_seed, _episode))
  seed = seed_before_mod % (2 ** 31 - 1)
  answer = {'_seed':_seed, '_episode':_episode, 'seed_before_mod':seed_before_mod, 'seed':seed}
  with open(f'hash_deterministic_{idx}.pckl', 'wb') as file:
    pickle.dump(answer, file)
  print(f'{answer=}')
  print(f'Done writing to hash_deterministic_{idx}.pckl')
else:
  file_range = (0,30)
  answers = []
  for idx in range(*file_range):
    with open(f'hash_deterministic_{idx}.pckl', 'rb') as file:
      answers.append(pickle.load(file))
  for answer in answers[1:]:
    if answers[0]['seed'] != answer['seed']:
      raise ValueError(f"{answers[0]['seed']} ≠ {answer['seed']} ... ({answers[0]['_seed']},{answers[0]['_episode']}), ({answer['_seed']},{answer['_episode']})")
  print('Everything is good')
