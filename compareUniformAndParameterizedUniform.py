import json
import matplotlib.pyplot as plt

class avgDict:
    def __init__(self):
        self.sums = {}
        self.counts = {}
    def __setitem__(self, key, value):
        self.sums[key] = self.sums.get(key, 0) + value
        self.counts[key] = self.counts.get(key, 0) + 1
    def __getitem__(self, key):
        return self.sums[key] / self.counts[key]
    def __contains__(self, key):
        return key in self.sums
    def sorted_keys(self):
        return sorted(self.sums)
    def items(self):
        return [(key,self[key]) for key in self.sorted_keys()]
    def keys(self):
        return list(self.sums.keys())
    def values(self):
        return [self[key] for key in self.sorted_keys()]
    def _interpolate_at(self, x):
        items = self.items()
        if x < items[0][0] or x > items[-1][0]:
            raise KeyError(f"Cannot get a value out of range.")
        if x in self:
            return self[x]
        for ix1,(x1,y1) in enumerate(items[:-1]):
            x2,y2 = items[ix1+1]
            if x1 < x and x < x2:
                return (y2-y1)/(x2-x1) * (x - x1) + y1
    def getValuesAt(self, X):
        return [self._interpolate_at(x) for x in X]

title = 'steps=1e6, replay_size=1e6'
filename = 'uniformVsParameterizedUniform'
logs = [
    {'path':'pendulum_normal', 'seeds':[0,10], 'name':'Uniform Dreamer', 'linestyle':'b'},
    {'path':'pendulum_parameterized_fifo_uniform_logits', 'seeds':[0,10], 'name':'Pendulum FiFo Uniform Logits', 'linestyle':'r'}
]

for modelrun in logs:
    path, name = modelrun['path'], modelrun['name']
    values = avgDict()
    for seed in range(modelrun['seeds'][0], modelrun['seeds'][1] + 1):
        with open(f'logs/{path}/{seed}/metrics.jsonl', 'r') as json_file:
            for line in json_file:
                json_dict = json.loads(line)
                if 'episode/score' not in json_dict or 'step' not in json_dict:
                    continue
                step = json_dict['step']
                score = json_dict['episode/score']
                values[step] = score
        print(f"Done seed {seed}")
    print(f'Done reading model \'{name}\' at path={path}')
    X = values.sorted_keys()
    Y = [values[x] for x in X]
    plt.plot(X, Y, modelrun['linestyle'], label=name)

plt.xlabel('Steps')
plt.ylabel('episode/score')
plt.legend()
plt.title(title)
plt.savefig(f'testFigures/{filename}.png')
