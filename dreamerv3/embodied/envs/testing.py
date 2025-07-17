import embodied
import numpy as np

class GridWorldDeterministicGoalWithEpsilon(embodied.Env):
    def __init__(self, size, seed, epsilon, obs_key='image'):
        self._size = size
        self._seed = seed
        self._random = np.random.default_rng(seed=self._seed)
        self._player_pos = None
        self._goal_pos = None
        self._done = True
        self._epsilon = epsilon
        self._obs_key = obs_key
        self._obs_space = {
            'reward': embodied.Space(np.float32, seed=self._seed),
            self._obs_key: embodied.Space(np.uint8, shape=(self._size,self._size,3), seed=self._seed),
            'is_first': embodied.Space(bool, seed=self._seed),
            'is_last': embodied.Space(bool, seed=self._seed),
            'is_terminal': embodied.Space(bool, seed=self._seed),
        }
        self._act_space = {
            'action': embodied.Space(np.int32, (), low=0, high=4, seed=self._seed),
            'reset': embodied.Space(bool, seed=self._seed)
        }

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def act_space(self):
        return self._act_space
    
    def step(self, action):
        if action['reset'] == True or self._done == True:
            self._done = False
            self._goal_pos = np.array([self._size - 1, self._size - 1], dtype=np.uint8)
            self._player_pos = np.array([0,0], dtype=np.uint8)
            return self._obs(0, is_first=True)
        self._move_player(action['action'])
        if np.array_equal(self._player_pos, self._goal_pos):
            self._done = True
            return self._obs(1, is_last=True, is_terminal=True)
        else:
            return self._obs(-0.01)
    
    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        return {
            self._obs_key: self.render(),
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal
        }
    
    def _move_player(self, action):
        if self._random.random() < self._epsilon:
            action = self._random.integers(0, 4) # epsilon
        if action == 0: # up
            self._player_pos += np.array([0,1], dtype=np.uint8)
        elif action == 1: # down
            self._player_pos += np.array([0,-1], dtype=np.uint8)
        elif action == 2: # left
            self._player_pos += np.array([-1,0], dtype=np.uint8)
        elif action == 3: # right
            self._player_pos += np.array([1,0], dtype=np.uint8)
        else:
            raise ValueError(f"Shouldn't get that action: {action}")
        self._player_pos[0] = min(max(self._player_pos[0], np.uint8(0)), np.uint8(self._size - 1))
        self._player_pos[1] = min(max(self._player_pos[1], np.uint8(0)), np.uint8(self._size - 1))
    
    def render(self):
        grid = np.full((self._size, self._size, 3), 255, dtype=np.uint8)
        grid[self._player_pos] = np.array([255, 0, 0], dtype=np.uint8)
        grid[self._goal_pos] = np.array([0, 255, 0])
        return grid

    def close(self):
        pass