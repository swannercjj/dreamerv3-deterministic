"""
This file contains simple environments imported over from useful-search-control for the use of debugging
and quick iterative development.
"""
import gymnasium
from dreamerv3.embodied.envs import from_gymnasium

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def make(environment_string, **kwargs):
    env_string = environment_string.replace('_','').replace(' ','').lower()
    if env_string == 'tmaze':
        env = TMaze(**kwargs)
    elif env_string == 'tworooms':
        env = TwoRooms(**kwargs)
    elif env_string in ('tworoomsfixed', 'tworoomsfixedgoal'):
        env = TwoRoomsFixedGoal(**kwargs)
    elif env_string in ('tworoomsfiftyfifty', 'tworoomsfiftyfiftygoal'):
        env = TwoRoomsFiftyFifty(**kwargs)
    else:
        raise NotImplementedError(f"Environment string not recognized: {environment_string} -> {env_string}")
    env = from_gymnasium.FromGymnasium(env, obs_key='vector')
    return env

class TMaze(gymnasium.Env):
    def __init__(self, reset_period=600, epsilon=0.05, seed=0):
        self.reset_period = reset_period
        self.seed = seed

        # PRNG for environment stochasticity.
        self._rand = np.random.RandomState(self.seed)
        self.epsilon = epsilon

        self.len_hallway = 5
        self.junction_height = 5
        self.num_states = 11
        self.terminating_states = [self.num_states - 2, self.num_states - 1]
        self.num_non_terminal_states = self.num_states - \
            len(self.terminating_states)
        self.num_actions = 4

        self.done_count = 0
        self.bonus_idx = 0
        self._init_topology()
        self.counts = np.zeros((self.num_states, self.num_actions))

        # pre-compute one_hot vectors to save time
        self.state_to_one_hot = {s: jax.nn.one_hot(s, self.num_states) for s in range(self.num_states)}
        self.action_to_one_hot = {a: jax.nn.one_hot(a, self.num_actions) for a in range(self.num_actions)}

        # to make it a gymnasium environment
        self.render_mode = "rgb_array"
        self.action_space = gymnasium.spaces.Discrete(self.num_actions)
        self.observation_space = gymnasium.spaces.Box(low=0., high=np.ones(self.num_states), dtype=np.float32)
        self.reward_range = (0,1)

    def render(self):
        # code adapted from:
        # https://stackoverflow.com/a/7821917
        fig = self.plot(np.zeros(self.num_states), keep_open=True)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def close(self):
        # Is there anything to do here?
        pass

    def _init_topology(self, initial_grid=None):
        """
                     10 (Terminal)
                      8
                      7
          0, 1, 2, 3, 4
                      5
                      6
                      9 (Terminal)
          Actions:
            0: north
            1: east
            2: south
            3: west
        """
        # Assign the adjacency
        self.adjacency_list = []
        self.adjacency_list.append([0, 1, 0, 0])  # 0
        self.adjacency_list.append([1, 2, 1, 0])  # 1
        self.adjacency_list.append([2, 3, 2, 1])  # 2
        self.adjacency_list.append([3, 4, 3, 2])  # 3
        self.adjacency_list.append([7, 4, 5, 3])  # 4
        self.adjacency_list.append([4, 5, 6, 5])  # 5
        self.adjacency_list.append([5, 6, 9, 6])  # 6
        self.adjacency_list.append([8, 7, 4, 7])  # 7
        self.adjacency_list.append([10, 8, 7, 8])  # 8
        self.adjacency_list.append([9, 9, 9, 9])  # 9
        self.adjacency_list.append([10, 10, 10, 10])  # 10

        self.state_xy = []
        # Vertical configuration.
        self.state_xy.append((0, -3))
        self.state_xy.append((0, -2))
        self.state_xy.append((0, -1))
        self.state_xy.append((0, 0))
        self.state_xy.append((0, 1))
        self.state_xy.append((1, 1))
        self.state_xy.append((2, 1))
        self.state_xy.append((-1, 1))
        self.state_xy.append((-2, 1))
        self.state_xy.append((3, 1))
        self.state_xy.append((-3, 1))

    def _reset_rewards(self):
        self.bonus_idx = (self.bonus_idx + 1) % 2

    def _get_obs(self):
        return self.state_to_one_hot[self.state]

    def _get_reward(self):
        if self.state == self.terminating_states[self.bonus_idx]:
            return 1.
        return 0.

    def _get_done(self):
        if self.state in self.terminating_states:
            self.done_count += 1
            return True
        return False

    def reset(self, *args, **kwargs):
        self.state = 0
        return self._get_obs(), {}

    def step(self, action):
        # action = np.argmax(action)
        # Reset the reward structure every reset_period episodes.
        if self.done_count == self.reset_period:
            self._reset_rewards()
            self.done_count = 0

        # Epsilon-stochastic actions.
        if self._rand.rand() < self.epsilon:
            action = self._rand.choice(self.num_actions)

        self.counts[self.state, action] += 1
        self.state = self.adjacency_list[self.state][action]
        return self._get_obs(), self._get_reward(), self._get_done(), False, {}

    def get_xy(self, state):
        (x, y) = self.state_xy[state]
        return x, y

    def get_state_number(self, x, y):
        for i, (u, v) in enumerate(self.state_xy):
            if (x, y) == (u, v):
                return i
        return None

    def plot(self, data, text=False, title="", static=False, save_dir=None, keep_open=False):
        import matplotlib.cm as cm
        import matplotlib.colors

        cmap = cm.Blues
        if static:
            norm = matplotlib.colors.Normalize(vmin=0.08, vmax=0.2, clip=False)
        else:
            norm = matplotlib.colors.Normalize(
                vmin=min(data), vmax=0.40, clip=True)
        # norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.6, clip=True)

        fig, ax = plt.subplots(figsize=(self.len_hallway, self.junction_height))
        # ax.set_title(title)
        ax.set_xlim((-2, 3))
        ax.set_ylim((-3, 2))
        ax.set_axis_off()

        # Create background.
        patch = mpl.patches.Rectangle((0, 0),
                                      self.len_hallway,
                                      self.junction_height,
                                      fill=True,
                                      color='white')
        ax.add_patch(patch)

        # Create maze cells.
        for s in range(self.num_non_terminal_states):
            x, y = self.get_xy(s)
            patch = mpl.patches.Rectangle((x, y), 1, 1,
                                          fill=True,
                                          facecolor=cmap(norm(data[s])),
                                          edgecolor='black',
                                          clip_on=False)
            ax.add_patch(patch)

            if text:
                text_color = "white" if data[s] > 0.3 else "black"
                ax.text(x + .5, y + .5, '%.2g' % data[s], va='center', ha='center', color=text_color)
        if save_dir:
            plt.savefig(f"{save_dir}/{title}.png")

        if not keep_open:
            plt.close()
        return fig

class TwoRooms(gymnasium.Env):
    """
    Attributes:
        available_goals: A list of coordinates that can be used as goal states.
        epsilon: Probability of an action failing.
            When an action fails, the agent will instead move in a random direction
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, seed=0, epsilon=0.05, goal_duration_steps=None, goal_duration_episodes=None):
        """
        Args:
            epsilon: Probability of an action failing.
                When an action fails, the resulting movement is randomly chosen
            goal_duration_steps: Number of steps taken before the goal state changes.
                The goal state changes immediately when this step count is reached, and
                can happen in the middle of an episode.
                Only one of `goal_duration_steps` and `goal_duration_episodes` can be set.
            goal_duration_episodes: Number of episodes completed before the goal state changes.
                Only one of `goal_duration_steps` and `goal_duration_episodes` can be set.
        """
        two_rooms_map = """
            xxxxxxxxxxxxxxx
            x      x      x
            x      x      x
            x             x
            x      x      x
            x      x      x
            x      x      x
            xxxxxxxxxxxxxxx"""

        self.directions = [
                np.array([-1,0]), #-y, up
                np.array([0,1]), #+x, right
                np.array([1,0]), #+y, down
                np.array([0,-1]) #-x, left
        ]

        # Process env map
        bool_map = []
        for row in two_rooms_map.split('\n')[1:]:
            bool_map.append([r==' ' for r in row.strip()])
        self.env_map = np.array(bool_map)
        self.available_goals = [[1,13], [6,13]] # top-right bottom-right
        # self.available_goals = [[1,13], [6,8]] # top-right bottom-left

        self.coords = []
        for y in range(self.env_map.shape[0]):
            for x in range(self.env_map.shape[1]):
                if [y,x] in self.available_goals:
                    continue
                elif self.env_map[y,x]: # If it's an open space
                    self.coords.append(np.array([y,x]))
        for goal in self.available_goals:
            self.coords.append(np.array(goal))

        self.num_states = len(self.coords)
        self.num_actions = 4

        # A list of state-index to one_hot_encoding used by the BSAdaptive agent.
        # NOTE: It is required that the terminal states are at the end of the state_to_one_hot
        # list so that agents do not sample them during planning.
        self.state_to_one_hot = [jax.nn.one_hot(s, self.num_states) for s in range(self.num_states)]
        self.action_to_one_hot = [jax.nn.one_hot(a, self.num_actions) for a in range(self.num_actions)]

        # A dictionary mapping from a coordinate tuple to a one-hot encoded state
        # Only the states which the agent can actually occupy are considered
        self.coord_to_one_hot_state = {}
        for index, coord in enumerate(self.coords):
            self.coord_to_one_hot_state[tuple(coord)] = self.state_to_one_hot[index]
        self.available_goals = [np.array(goal) for goal in self.available_goals] # turn these into numpy arrays for saving

        self.counts = np.zeros((self.num_states, self.num_actions))

        # Process other params
        self.epsilon = epsilon
        self.reward_rage = (0,1)
        self.action_space = gymnasium.spaces.Discrete(self.num_actions)
        self.observation_space = gymnasium.spaces.Box(low=0., high=np.ones(self.num_states), dtype=np.float32)
        self.render_mode = "rgb_array"

        if goal_duration_steps is None and goal_duration_episodes is None:
            goal_duration_episodes = 1
        elif goal_duration_steps is not None and goal_duration_episodes is not None:
            raise ValueError('Both goal_duration_steps and goal_duration_episodes were assigned values. Only one can be used at a time.')
        self.goal_duration_steps = goal_duration_steps
        self.goal_duration_episodes = goal_duration_episodes
        self.step_count = 0
        self.episode_count = 0

        self.pos = None
        self.goal = None

        self.seed(seed)

    def _get_obs(self):
        """
        Return an observation which is a one-hot encoded state. 
        The one-hot encoding corresponds to the number of the state within the total 
        number of valid states. 
        E.g. 4,5 might be the 20th valid state and so this will produce a 
        vector with a one only at index 19. 
        """
        return self.coord_to_one_hot_state[tuple([self.pos[0], self.pos[1]])]

    def step(self, action):
        # Update state
        if self.rand.rand() < self.epsilon:
            action_idx = self.rand.choice(self.num_actions)
        else:
            action_idx = action
        p = self.pos + self.directions[action_idx]
        if self.env_map[p[0],p[1]]:
            self.pos = p
        # obs = np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])
        obs = self._get_obs()
        
        self.counts[np.argmax(obs), action_idx] += 1
        
        # If the agent's position == any of the goals
        if tuple(self.pos) in {tuple(goal) for goal in self.available_goals}:
            if (self.pos == self.goal).all():
                # if the agent is at the actual goal
                reward = 1
            else:
                # if the agent is at a different goal
                reward = 0
            done = True
            self.pos = None
        else:
            reward = 0
            done = False
        # Update counts
        if self.goal_duration_steps is not None:
            self.step_count += 1
            if self.step_count >= self.goal_duration_steps:
                self.step_count = 0
                self.reset_goal()
        # Return updated states
        return obs, reward, done, False, {}

    def reset(self, *args, **kwargs):
        self.reset_pos()
        # Check if goal state needs to be changed
        if self.goal is None:
            self.reset_goal()
        if self.goal_duration_steps is None and self.goal_duration_episodes is None:
            self.reset_goal()
        elif self.goal_duration_episodes is not None:
            if self.episode_count >= self.goal_duration_episodes:
                self.reset_goal()
                self.episode_count = 0
            self.episode_count += 1
        # return np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])
        return self._get_obs(), {}

    def reset_goal(self, goal=None):
        if goal is not None:
            self.goal = goal[:]
        else:
            if (self.goal == self.available_goals[0]).all():
                self.goal = self.available_goals[1][:]
            else:
                self.goal = self.available_goals[0][:]

    def reset_pos(self):
        # if self.goal is None:
        #     pos_index = self.rand.randint(0,len(self.coords))
        # else:
        #     pos_index = self.rand.randint(0,len(self.coords)-1)
        #     if (self.coords[pos_index] == self.goal).all():
        #         pos_index = len(self.coords)-1
        self.pos = np.array([6,1])

    def seed(self, seed=None):
        self.rand = np.random.RandomState(seed)
        self.action_space._np_random = self.rand

    def render(self):
        # code adapted from:
        # https://stackoverflow.com/a/7821917
        fig = self.plot(np.zeros(self.num_states), keep_open=True)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def close(self):
        self.pos = None
        self.goal = None

    def plot(self, data, text=False, title="", save_title="", save_dir=None, keep_open=False, outline_goals=False):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors

        cmap = cm.Blues
        norm = matplotlib.colors.Normalize(
            vmin=np.min(data), vmax=np.max(data), clip=True)

        fig, ax = plt.subplots(figsize=(self.env_map.shape[1], self.env_map.shape[0]+1))
        if len(title) > 0:
            ax.set_title(title)
        ax.set_xlim((0, self.env_map.shape[1]))
        ax.set_ylim((0, self.env_map.shape[0]))
        ax.set_axis_off()

        # Create background.
        patch = mpl.patches.Rectangle((0, 0),
                                        self.env_map.shape[1],
                                        self.env_map.shape[0],
                                        fill=True,
                                        color='white')
        ax.add_patch(patch)

        # Create maze cells.
        # print(f'{data.shape=}')
        for s, coord in enumerate(self.coords):
            # print(f'{data[s].shape}=')
            y, x = coord

            value = data[s] if s < len(data) else 0.
            color = cmap(norm(value))
            patch = mpl.patches.Rectangle((x, y), 1, 1,
                                        fill=True,
                                        facecolor=color,
                                        edgecolor='black',
                                        clip_on=False)
            ax.add_patch(patch)
            if text:
                text_color = "white" if value > 0.3 else "black"
                ax.text(x + .5, y + .5, '%.2g' % value, va='center', ha='center', color=text_color)

        if outline_goals:
            for coord in self.available_goals:
                y, x = coord
                patch = mpl.patches.Rectangle((x,y), 1, 1, edgecolor='red', fill=False, linewidth=2)
                ax.add_patch(patch)

        plt.gca().invert_yaxis() # to account for coordinates being top=0 bottom=+
        if save_dir:
            plt.savefig(f"{save_dir}/{save_title}.png")

        if not keep_open:
            plt.close()

    def plot_q_values(self, q, text=False, title="", save_title="", save_dir=None, keep_open=False, outline_goals=False):
        import matplotlib as mpl
        import matplotlib.cm as cm
        import matplotlib.colors
        import matplotlib.pyplot as plt

        cmap = cm.Blues_r
        norm = matplotlib.colors.Normalize(
            vmin=np.min(q), vmax=np.max(q), clip=True)

        fig, ax = plt.subplots(figsize=(self.env_map.shape[1], self.env_map.shape[0]+1))
        if len(title) > 0:
            ax.set_title(title)
        ax.set_xlim((0, self.env_map.shape[1]))
        ax.set_ylim((0, self.env_map.shape[0]))
        ax.set_axis_off()

        # Create background.
        patch = mpl.patches.Rectangle((0, 0),
                                        self.env_map.shape[1],
                                        self.env_map.shape[0],
                                        fill=True,
                                        color='white')
        ax.add_patch(patch)

        # Create maze cells.
        for s, coord in enumerate(self.coords):
            y, x = coord

            # left action
            xy = np.array([[x, y], [x + 0.5, y + 0.5], [x, y + 1]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[3, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.25, y+.5, '%.2g' % q[3, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # down action
            xy = np.array([[x, y + 1], [x + 0.5, y + 0.5], [x + 1, y + 1]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[2, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.5, y+.75, '%.2g' % q[2, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # right action
            xy = np.array([[x + 0.5, y + 0.5], [x + 1, y + 1], [x + 1, y]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[1, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.75, y+.5, '%.2g' % q[1, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # up action
            xy = np.array([[x, y], [x + 0.5, y + 0.5], [x + 1, y]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[0, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.5, y+.25, '%.2g' % q[0, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

        if outline_goals:
            for coord in self.available_goals:
                y, x = coord
                patch = mpl.patches.Rectangle((x,y), 1, 1, edgecolor='red', fill=False, linewidth=2)
                ax.add_patch(patch)

        plt.gca().invert_yaxis() # to account for coordinates being top=0 bottom=+
        if save_dir:
            plt.savefig(f"{save_dir}/{save_title}.png")

        if not keep_open:
            plt.close()

class TwoRoomsFixedGoal(gymnasium.Env):
    """
    Attributes:
        available_goals: A list of coordinates that can be used as goal states.
        epsilon: Probability of an action failing.
            When an action fails, the agent will instead move in a random direction
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, seed=0, epsilon=0.05, goal_duration_steps=None, goal_duration_episodes=None):
        """
        Args:
            epsilon: Probability of an action failing.
                When an action fails, the resulting movement is randomly chosen
            goal_duration_steps: Number of steps taken before the goal state changes.
                The goal state changes immediately when this step count is reached, and
                can happen in the middle of an episode.
                Only one of `goal_duration_steps` and `goal_duration_episodes` can be set.
            goal_duration_episodes: Number of episodes completed before the goal state changes.
                Only one of `goal_duration_steps` and `goal_duration_episodes` can be set.
        """
        two_rooms_map = """
            xxxxxxxxxxxxxxx
            x      x      x
            x      x      x
            x             x
            x      x      x
            x      x      x
            x      x      x
            xxxxxxxxxxxxxxx"""

        self.directions = [
                np.array([-1,0]), #-y, up
                np.array([0,1]), #+x, right
                np.array([1,0]), #+y, down
                np.array([0,-1]) #-x, left
        ]

        # Process env map
        bool_map = []
        for row in two_rooms_map.split('\n')[1:]:
            bool_map.append([r==' ' for r in row.strip()])
        self.env_map = np.array(bool_map)
        self.available_goals = [[1,13], [6,13]] # top-right bottom-right
        # self.available_goals = [[1,13], [6,8]] # top-right bottom-left

        self.coords = []
        for y in range(self.env_map.shape[0]):
            for x in range(self.env_map.shape[1]):
                if [y,x] in self.available_goals:
                    continue
                elif self.env_map[y,x]: # If it's an open space
                    self.coords.append(np.array([y,x]))
        for goal in self.available_goals:
            self.coords.append(np.array(goal))

        self.num_states = len(self.coords)
        self.num_actions = 4

        # A list of state-index to one_hot_encoding used by the BSAdaptive agent.
        # NOTE: It is required that the terminal states are at the end of the state_to_one_hot
        # list so that agents do not sample them during planning.
        self.state_to_one_hot = [jax.nn.one_hot(s, self.num_states) for s in range(self.num_states)]
        self.action_to_one_hot = [jax.nn.one_hot(a, self.num_actions) for a in range(self.num_actions)]

        # A dictionary mapping from a coordinate tuple to a one-hot encoded state
        # Only the states which the agent can actually occupy are considered
        self.coord_to_one_hot_state = {}
        for index, coord in enumerate(self.coords):
            self.coord_to_one_hot_state[tuple(coord)] = self.state_to_one_hot[index]
        self.available_goals = [np.array(goal) for goal in self.available_goals] # turn these into numpy arrays for saving

        self.counts = np.zeros((self.num_states, self.num_actions))

        # Process other params
        self.epsilon = epsilon
        self.reward_rage = (0,1)
        self.action_space = gymnasium.spaces.Discrete(self.num_actions)
        self.observation_space = gymnasium.spaces.Box(low=0., high=np.ones(self.num_states), dtype=np.float32)
        self.render_mode = "rgb_array"

        if goal_duration_steps is None and goal_duration_episodes is None:
            goal_duration_episodes = 1
        elif goal_duration_steps is not None and goal_duration_episodes is not None:
            raise ValueError('Both goal_duration_steps and goal_duration_episodes were assigned values. Only one can be used at a time.')
        self.goal_duration_steps = goal_duration_steps
        self.goal_duration_episodes = goal_duration_episodes
        self.step_count = 0
        self.episode_count = 0

        self.pos = None
        self.goal = None

        self.seed(seed)

    def _get_obs(self):
        """
        Return an observation which is a one-hot encoded state. 
        The one-hot encoding corresponds to the number of the state within the total 
        number of valid states. 
        E.g. 4,5 might be the 20th valid state and so this will produce a 
        vector with a one only at index 19. 
        """
        return self.coord_to_one_hot_state[tuple([self.pos[0], self.pos[1]])]

    def step(self, action):
        # Update state
        if self.rand.rand() < self.epsilon:
            action_idx = self.rand.choice(self.num_actions)
        else:
            action_idx = action
        p = self.pos + self.directions[action_idx]
        if self.env_map[p[0],p[1]]:
            self.pos = p
        # obs = np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])
        obs = self._get_obs()
        
        self.counts[np.argmax(obs), action_idx] += 1
        
        # If the agent's position == any of the goals
        if tuple(self.pos) in {tuple(goal) for goal in self.available_goals}:
            if (self.pos == self.goal).all():
                # if the agent is at the actual goal
                reward = 1
            else:
                # if the agent is at a different goal
                reward = 0
            done = True
            self.pos = None
        else:
            reward = 0
            done = False
        # Update counts
        if self.goal_duration_steps is not None:
            self.step_count += 1
            if self.step_count >= self.goal_duration_steps:
                self.step_count = 0
                self.reset_goal()
        # Return updated states
        return obs, reward, done, False, {}

    def reset(self, *args, **kwargs):
        self.reset_pos()
        # Check if goal state needs to be changed
        if self.goal is None:
            self.reset_goal()
        if self.goal_duration_steps is None and self.goal_duration_episodes is None:
            self.reset_goal()
        elif self.goal_duration_episodes is not None:
            if self.episode_count >= self.goal_duration_episodes:
                self.reset_goal()
                self.episode_count = 0
            self.episode_count += 1
        # return np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])
        return self._get_obs(), {}

    def reset_goal(self, goal=None):
        if goal is not None:
            self.goal = goal[:]
        else:
            self.goal = self.available_goals[1][:]

    def reset_pos(self):
        # if self.goal is None:
        #     pos_index = self.rand.randint(0,len(self.coords))
        # else:
        #     pos_index = self.rand.randint(0,len(self.coords)-1)
        #     if (self.coords[pos_index] == self.goal).all():
        #         pos_index = len(self.coords)-1
        self.pos = np.array([6,1])

    def seed(self, seed=None):
        self.rand = np.random.RandomState(seed)
        self.action_space._np_random = self.rand

    def render(self):
        # code adapted from:
        # https://stackoverflow.com/a/7821917
        fig = self.plot(np.zeros(self.num_states), keep_open=True)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def close(self):
        self.pos = None
        self.goal = None

    def plot(self, data, text=False, title="", save_title="", save_dir=None, keep_open=False, outline_goals=False):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors

        cmap = cm.Blues
        norm = matplotlib.colors.Normalize(
            vmin=np.min(data), vmax=np.max(data), clip=True)

        fig, ax = plt.subplots(figsize=(self.env_map.shape[1], self.env_map.shape[0]+1))
        if len(title) > 0:
            ax.set_title(title)
        ax.set_xlim((0, self.env_map.shape[1]))
        ax.set_ylim((0, self.env_map.shape[0]))
        ax.set_axis_off()

        # Create background.
        patch = mpl.patches.Rectangle((0, 0),
                                        self.env_map.shape[1],
                                        self.env_map.shape[0],
                                        fill=True,
                                        color='white')
        ax.add_patch(patch)

        # Create maze cells.
        # print(f'{data.shape=}')
        for s, coord in enumerate(self.coords):
            # print(f'{data[s].shape}=')
            y, x = coord

            value = data[s] if s < len(data) else 0.
            color = cmap(norm(value))
            patch = mpl.patches.Rectangle((x, y), 1, 1,
                                        fill=True,
                                        facecolor=color,
                                        edgecolor='black',
                                        clip_on=False)
            ax.add_patch(patch)
            if text:
                text_color = "white" if value > 0.3 else "black"
                ax.text(x + .5, y + .5, '%.2g' % value, va='center', ha='center', color=text_color)

        if outline_goals:
            for coord in self.available_goals:
                y, x = coord
                patch = mpl.patches.Rectangle((x,y), 1, 1, edgecolor='red', fill=False, linewidth=2)
                ax.add_patch(patch)

        plt.gca().invert_yaxis() # to account for coordinates being top=0 bottom=+
        if save_dir:
            plt.savefig(f"{save_dir}/{save_title}.png")

        if not keep_open:
            plt.close()

    def plot_q_values(self, q, text=False, title="", save_title="", save_dir=None, keep_open=False, outline_goals=False):
        import matplotlib as mpl
        import matplotlib.cm as cm
        import matplotlib.colors
        import matplotlib.pyplot as plt

        cmap = cm.Blues_r
        norm = matplotlib.colors.Normalize(
            vmin=np.min(q), vmax=np.max(q), clip=True)

        fig, ax = plt.subplots(figsize=(self.env_map.shape[1], self.env_map.shape[0]+1))
        if len(title) > 0:
            ax.set_title(title)
        ax.set_xlim((0, self.env_map.shape[1]))
        ax.set_ylim((0, self.env_map.shape[0]))
        ax.set_axis_off()

        # Create background.
        patch = mpl.patches.Rectangle((0, 0),
                                        self.env_map.shape[1],
                                        self.env_map.shape[0],
                                        fill=True,
                                        color='white')
        ax.add_patch(patch)

        # Create maze cells.
        for s, coord in enumerate(self.coords):
            y, x = coord

            # left action
            xy = np.array([[x, y], [x + 0.5, y + 0.5], [x, y + 1]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[3, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.25, y+.5, '%.2g' % q[3, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # down action
            xy = np.array([[x, y + 1], [x + 0.5, y + 0.5], [x + 1, y + 1]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[2, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.5, y+.75, '%.2g' % q[2, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # right action
            xy = np.array([[x + 0.5, y + 0.5], [x + 1, y + 1], [x + 1, y]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[1, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.75, y+.5, '%.2g' % q[1, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # up action
            xy = np.array([[x, y], [x + 0.5, y + 0.5], [x + 1, y]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[0, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.5, y+.25, '%.2g' % q[0, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

        if outline_goals:
            for coord in self.available_goals:
                y, x = coord
                patch = mpl.patches.Rectangle((x,y), 1, 1, edgecolor='red', fill=False, linewidth=2)
                ax.add_patch(patch)

        plt.gca().invert_yaxis() # to account for coordinates being top=0 bottom=+
        if save_dir:
            plt.savefig(f"{save_dir}/{save_title}.png")

        if not keep_open:
            plt.close()

class TwoRoomsFiftyFifty(gymnasium.Env):
    """
    Attributes:
        available_goals: A list of coordinates that can be used as goal states.
        epsilon: Probability of an action failing.
            When an action fails, the agent will instead move in a random direction
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, seed=0, epsilon=0.05, goal_duration_steps=None, goal_duration_episodes=None):
        """
        Args:
            epsilon: Probability of an action failing.
                When an action fails, the resulting movement is randomly chosen
            goal_duration_steps: Number of steps taken before the goal state changes.
                The goal state changes immediately when this step count is reached, and
                can happen in the middle of an episode.
                Only one of `goal_duration_steps` and `goal_duration_episodes` can be set.
            goal_duration_episodes: Number of episodes completed before the goal state changes.
                Only one of `goal_duration_steps` and `goal_duration_episodes` can be set.
        """
        two_rooms_map = """
            xxxxxxxxxxxxxxx
            x      x      x
            x      x      x
            x             x
            x      x      x
            x      x      x
            x      x      x
            xxxxxxxxxxxxxxx"""

        self.directions = [
                np.array([-1,0]), #-y, up
                np.array([0,1]), #+x, right
                np.array([1,0]), #+y, down
                np.array([0,-1]) #-x, left
        ]

        # Process env map
        bool_map = []
        for row in two_rooms_map.split('\n')[1:]:
            bool_map.append([r==' ' for r in row.strip()])
        self.env_map = np.array(bool_map)
        self.available_goals = [[1,13], [6,13]] # top-right bottom-right
        # self.available_goals = [[1,13], [6,8]] # top-right bottom-left

        self.coords = []
        for y in range(self.env_map.shape[0]):
            for x in range(self.env_map.shape[1]):
                if [y,x] in self.available_goals:
                    continue
                elif self.env_map[y,x]: # If it's an open space
                    self.coords.append(np.array([y,x]))
        for goal in self.available_goals:
            self.coords.append(np.array(goal))

        self.num_states = len(self.coords)
        self.num_actions = 4

        # A list of state-index to one_hot_encoding used by the BSAdaptive agent.
        # NOTE: It is required that the terminal states are at the end of the state_to_one_hot
        # list so that agents do not sample them during planning.
        self.state_to_one_hot = [jax.nn.one_hot(s, self.num_states) for s in range(self.num_states)]
        self.action_to_one_hot = [jax.nn.one_hot(a, self.num_actions) for a in range(self.num_actions)]

        # A dictionary mapping from a coordinate tuple to a one-hot encoded state
        # Only the states which the agent can actually occupy are considered
        self.coord_to_one_hot_state = {}
        for index, coord in enumerate(self.coords):
            self.coord_to_one_hot_state[tuple(coord)] = self.state_to_one_hot[index]
        self.available_goals = [np.array(goal) for goal in self.available_goals] # turn these into numpy arrays for saving

        self.counts = np.zeros((self.num_states, self.num_actions))

        # Process other params
        self.epsilon = epsilon
        self.reward_rage = (0,1)
        self.action_space = gymnasium.spaces.Discrete(self.num_actions)
        self.observation_space = gymnasium.spaces.Box(low=0., high=np.ones(self.num_states), dtype=np.float32)
        self.render_mode = "rgb_array"

        if goal_duration_steps is None and goal_duration_episodes is None:
            goal_duration_episodes = 1
        elif goal_duration_steps is not None and goal_duration_episodes is not None:
            raise ValueError('Both goal_duration_steps and goal_duration_episodes were assigned values. Only one can be used at a time.')
        self.goal_duration_steps = goal_duration_steps
        self.goal_duration_episodes = goal_duration_episodes
        self.step_count = 0
        self.episode_count = 0

        self.pos = None
        self.goal = None

        self.seed(seed)

    def _get_obs(self):
        """
        Return an observation which is a one-hot encoded state. 
        The one-hot encoding corresponds to the number of the state within the total 
        number of valid states. 
        E.g. 4,5 might be the 20th valid state and so this will produce a 
        vector with a one only at index 19. 
        """
        return self.coord_to_one_hot_state[tuple([self.pos[0], self.pos[1]])]

    def step(self, action):
        # Update state
        if self.rand.rand() < self.epsilon:
            action_idx = self.rand.choice(self.num_actions)
        else:
            action_idx = action
        p = self.pos + self.directions[action_idx]
        if self.env_map[p[0],p[1]]:
            self.pos = p
        # obs = np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])
        obs = self._get_obs()
        
        self.counts[np.argmax(obs), action_idx] += 1
        
        # If the agent's position == any of the goals
        if tuple(self.pos) in {tuple(goal) for goal in self.available_goals}:
            reward = self.rand.choice([0,1])
            done = True
            self.pos = None
        else:
            reward = 0
            done = False
        # Update counts
        if self.goal_duration_steps is not None:
            self.step_count += 1
            if self.step_count >= self.goal_duration_steps:
                self.step_count = 0
                self.reset_goal()
        # Return updated states
        return obs, reward, done, False, {}

    def reset(self, *args, **kwargs):
        self.reset_pos()
        # Check if goal state needs to be changed
        if self.goal is None:
            self.reset_goal()
        if self.goal_duration_steps is None and self.goal_duration_episodes is None:
            self.reset_goal()
        elif self.goal_duration_episodes is not None:
            if self.episode_count >= self.goal_duration_episodes:
                self.reset_goal()
                self.episode_count = 0
            self.episode_count += 1
        # return np.array([self.pos[0],self.pos[1],self.goal[0],self.goal[1]])
        return self._get_obs(), {}

    def reset_goal(self, goal=None):
        if goal is not None:
            self.goal = goal[:]
        else:
            pass

    def reset_pos(self):
        # if self.goal is None:
        #     pos_index = self.rand.randint(0,len(self.coords))
        # else:
        #     pos_index = self.rand.randint(0,len(self.coords)-1)
        #     if (self.coords[pos_index] == self.goal).all():
        #         pos_index = len(self.coords)-1
        self.pos = np.array([6,1])

    def seed(self, seed=None):
        self.rand = np.random.RandomState(seed)
        self.action_space._np_random = self.rand

    def render(self):
        # code adapted from:
        # https://stackoverflow.com/a/7821917
        fig = self.plot(np.zeros(self.num_states), keep_open=True)

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def close(self):
        self.pos = None
        self.goal = None

    def plot(self, data, text=False, title="", save_title="", save_dir=None, keep_open=False, outline_goals=False):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors

        cmap = cm.Blues
        norm = matplotlib.colors.Normalize(
            vmin=np.min(data), vmax=np.max(data), clip=True)

        fig, ax = plt.subplots(figsize=(self.env_map.shape[1], self.env_map.shape[0]+1))
        if len(title) > 0:
            ax.set_title(title)
        ax.set_xlim((0, self.env_map.shape[1]))
        ax.set_ylim((0, self.env_map.shape[0]))
        ax.set_axis_off()

        # Create background.
        patch = mpl.patches.Rectangle((0, 0),
                                        self.env_map.shape[1],
                                        self.env_map.shape[0],
                                        fill=True,
                                        color='white')
        ax.add_patch(patch)

        # Create maze cells.
        # print(f'{data.shape=}')
        for s, coord in enumerate(self.coords):
            # print(f'{data[s].shape}=')
            y, x = coord

            value = data[s] if s < len(data) else 0.
            color = cmap(norm(value))
            patch = mpl.patches.Rectangle((x, y), 1, 1,
                                        fill=True,
                                        facecolor=color,
                                        edgecolor='black',
                                        clip_on=False)
            ax.add_patch(patch)
            if text:
                text_color = "white" if value > 0.3 else "black"
                ax.text(x + .5, y + .5, '%.2g' % value, va='center', ha='center', color=text_color)

        if outline_goals:
            for coord in self.available_goals:
                y, x = coord
                patch = mpl.patches.Rectangle((x,y), 1, 1, edgecolor='red', fill=False, linewidth=2)
                ax.add_patch(patch)

        plt.gca().invert_yaxis() # to account for coordinates being top=0 bottom=+
        if save_dir:
            plt.savefig(f"{save_dir}/{save_title}.png")

        if not keep_open:
            plt.close()

    def plot_q_values(self, q, text=False, title="", save_title="", save_dir=None, keep_open=False, outline_goals=False):
        import matplotlib as mpl
        import matplotlib.cm as cm
        import matplotlib.colors
        import matplotlib.pyplot as plt

        cmap = cm.Blues_r
        norm = matplotlib.colors.Normalize(
            vmin=np.min(q), vmax=np.max(q), clip=True)

        fig, ax = plt.subplots(figsize=(self.env_map.shape[1], self.env_map.shape[0]+1))
        if len(title) > 0:
            ax.set_title(title)
        ax.set_xlim((0, self.env_map.shape[1]))
        ax.set_ylim((0, self.env_map.shape[0]))
        ax.set_axis_off()

        # Create background.
        patch = mpl.patches.Rectangle((0, 0),
                                        self.env_map.shape[1],
                                        self.env_map.shape[0],
                                        fill=True,
                                        color='white')
        ax.add_patch(patch)

        # Create maze cells.
        for s, coord in enumerate(self.coords):
            y, x = coord

            # left action
            xy = np.array([[x, y], [x + 0.5, y + 0.5], [x, y + 1]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[3, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.25, y+.5, '%.2g' % q[3, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # down action
            xy = np.array([[x, y + 1], [x + 0.5, y + 0.5], [x + 1, y + 1]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[2, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.5, y+.75, '%.2g' % q[2, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # right action
            xy = np.array([[x + 0.5, y + 0.5], [x + 1, y + 1], [x + 1, y]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[1, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.75, y+.5, '%.2g' % q[1, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

            # up action
            xy = np.array([[x, y], [x + 0.5, y + 0.5], [x + 1, y]])
            patch = mpl.patches.Polygon(xy, 
                                        facecolor=cmap(norm(q[0, s])),
                                        edgecolor='black')
            if text:
                ax.text(x+.5, y+.25, '%.2g' % q[0, s], va='center', ha='center', fontsize=8)
            ax.add_patch(patch)

        if outline_goals:
            for coord in self.available_goals:
                y, x = coord
                patch = mpl.patches.Rectangle((x,y), 1, 1, edgecolor='red', fill=False, linewidth=2)
                ax.add_patch(patch)

        plt.gca().invert_yaxis() # to account for coordinates being top=0 bottom=+
        if save_dir:
            plt.savefig(f"{save_dir}/{save_title}.png")

        if not keep_open:
            plt.close()
