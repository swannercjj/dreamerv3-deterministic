import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import stats
from pathlib import Path


def create_output_directory():
    global OUTPUT_DIRECTORY
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

def mean_confidence_interval(data, confidence=0.95):
    """
    Code obtained from the link below:
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
    h = se * stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, se, h

def get_rewards(agent_dict, seed):
    x = []
    y = []
    with open(f"{agent_dict['log_dir']}/{seed}/metrics.jsonl", 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        line = json.loads(json_str)
        if 'episode/score' in line and 'step' in line:
            x.append(line['step'])
            y.append(line['episode/score'])
    return x, y

# def average_with_interpolate(data_pairs):
#     def get_value_at(x, data_pair):
#         if x <= 0:
#             return 0
#         X, Y = data_pair
#         if x <= X[0]:
#             x0, y0 = 0, 0
#             x1, y1 = X[0], Y[0]
#             return (y1 - y0) / (x1 - x0) * (x - x0) + y0
#         if x >= X[-1]:
#             x0, y0 = X[-2], Y[-2]
#             x1, y1 = X[-1], Y[-1]
#             return (y1 - y0) / (x1 - x0) * (x - x0) + y0
#         for idx in range(len(X)-1):
#             x0, y0 = X[idx], Y[idx]
#             x1, y1 = X[idx+1], Y[idx+1]
#             if x0 <= x and x < x1:
#                 return (y1 - y0) / (x1 - x0) * (x - x0) + y0
#         raise ValueError(f"How did you input a value we don't recognize? x={x}")
#     significant_x = set()
#     for X,Y in data_pairs:
#         for x in X:
#             significant_x.add(x)
#     significant_x = sorted(significant_x)
#     Y = []
#     for x in significant_x:
#         ys = [get_value_at(x, data_pair) for data_pair in data_pairs]
#         Y.append(np.mean(ys))
#     return [0] + significant_x, [0] + Y

def average_with_interpolate(data_pairs):
    def equation(x, x0, x1, y0, y1):
        return (y1 - y0) / (x1 - x0) * (x - x0) + y0
    indices = [0 for _ in range(len(data_pairs))]
    X, Y = [0], [0]
    while any([idx < len(pair[0]) for idx,pair in zip(indices, data_pairs)]):
        next_min = min(pair[0][i] if i < len(pair[0]) else float('inf') for pair,i in zip(data_pairs, indices))
        y_values = []
        for pair_index in range(len(data_pairs)):
            pair = data_pairs[pair_index]
            i = indices[pair_index]
            y = None
            if i >= len(pair[0]):
                y = equation(next_min, pair[0][-2], pair[0][-1], pair[1][-2], pair[1][-1])
            elif pair[0][i] == next_min:
                y = pair[1][i]
                indices[pair_index] += 1
            elif i == 0:
                y = equation(next_min, 0, pair[0][0], 0, pair[1][0])
            else:
                y = equation(next_min, pair[0][i-1], pair[0][i], pair[1][i-1], pair[1][i])
            y_values.append(y)
        X.append(next_min)
        Y.append(np.mean(y_values))
    return X, Y


def get_average_total_reward(agent_dict):
    print(f"Getting total reward data for {agent_dict['name']} ...")
    total_rewards = []
    for seed in range(*agent_dict['array_range']):
        _, y = get_rewards(agent_dict, seed)
        total_rewards.append(np.sum(y))
    m, _, conf = mean_confidence_interval(total_rewards)
    print(f"Finished getting total reward data for {agent_dict['name']}.")
    return m, conf

def plot_total_reward_bar_graph(total_rewards, output_path='barplot.png', colors=None):
    print(f'Plotting total reward bar graph to {output_path} ...')
    names = [d['name'] for d in total_rewards]
    heights = [d['total_reward'] for d in total_rewards]
    conf_intervals = [d['err'] for d in total_rewards]
    plt.bar(names, heights, yerr=conf_intervals, capsize=10, color=colors, edgecolor='black')
    plt.ylabel('Total Reward')
    plt.savefig(Path(OUTPUT_DIRECTORY) / output_path)
    plt.clf()
    print(f"Finished plotting total reward bar graph.")

def get_average_total_reward_cumsum(agent_dict):
    print(f"Getting cumulative reward cumsum data for {agent_dict['name']} ...")
    data_pairs = [get_rewards(agent_dict, seed) for seed in range(*agent_dict['array_range'])]
    data_pairs = [(x, np.cumsum(y)) for x,y in data_pairs]
    print(f"\tInterpolating the values to average...")
    x, y = average_with_interpolate(data_pairs)
    print(f"Finished getting cumulative reward data for {agent_dict['name']}.")
    return x, y

def plot_total_reward_line_graph(total_rewards, output_path='total_reward_plot.png', colors=None):
    print(f'Plotting total reward line graph to {output_path} ...')
    for i,agent in enumerate(total_rewards):
        plt.plot(agent['steps'], agent['total_reward'], label=agent['name'], color=colors[i], markeredgecolor='black')
    plt.legend()
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Timestep')
    plt.savefig(Path(OUTPUT_DIRECTORY) / output_path)
    plt.clf()
    print(f"Finished plotting total reward line graph.")

def calc_average_rewards(x, tr_cumsum, type_of_average='cumsum'):
    if type_of_average == 'cumsum':
        avg_rewards = tr_cumsum / np.arange(1, len(tr_cumsum) + 1)
    elif type_of_average == 'derivative':
        derivatives = [0]
        for i in range(1,len(tr_cumsum)):
            x0, y0 = x[i-1], tr_cumsum[i-1]
            x1, y1 = x[i], tr_cumsum[i]
            derivatives.append((y1 - y0) / (x1 - x0))
        avg_rewards = derivatives
    else:
        raise ValueError(f"Unknown type of average reward calculation.")
    return x, avg_rewards

def plot_average_reward_graph(average_rewards, output_path='avgreward.png', colors=None):
    print(f'Plotting average reward graph to {output_path} ...')
    for i,agent in enumerate(average_rewards):
        plt.plot(agent['steps'], agent['average_reward'], label=agent['name'], color=colors[i], markeredgecolor='black')
    plt.legend()
    plt.ylabel('Average Reward')
    plt.xlabel('Timestep')
    plt.savefig(Path(OUTPUT_DIRECTORY) / output_path)
    plt.clf()
    print(f"Finished plotting average reward graph.")

def get_episode_lengths(agent_dict):
    print(f"Getting episode length data for {agent_dict['name']} ...")
    lengths = []
    for seed in range(*agent_dict['array_range']):
        with open(f"{agent_dict['log_dir']}/{seed}/metrics.jsonl", 'r') as json_file:
            json_list = list(json_file)
        episode = 0
        for json_str in json_list:
            line = json.loads(json_str)
            if 'episode/length' in line:
                episode_length = line['episode/length']
                if len(lengths) <= episode:
                    lengths.append([episode_length])
                else:
                    lengths[episode].append(episode_length)
                episode += 1
    m = []
    conf = []
    for episode,episode_lengths in enumerate(lengths):
        avg, _, intvl = mean_confidence_interval(episode_lengths)
        m.append(avg)
        conf.append(intvl)
    print(f"Finished getting episode length data for {agent_dict['name']}.")
    return m, conf

def plot_episode_lengths(episode_lengths, output_path='episode_lengths.png', colors=None):
    print(f'Plotting episode length graph to {output_path} ...')
    for i,agent in enumerate(episode_lengths):
        plt.plot(list(range(1, len(agent['episode_lengths']) + 1)), agent['episode_lengths'], label=agent['name'], color=colors[i], markeredgecolor='black')
    plt.legend()
    plt.ylabel('Episode Length')
    plt.xlabel('Episode')
    plt.savefig(Path(OUTPUT_DIRECTORY) / output_path)
    plt.clf()
    print(f"Finished plotting episode length graph.")


OUTPUT_DIRECTORY = './plots/movingVsStationaryGoalTwoRooms/'
agents = [
    {
        'name': 'Switching Goals',
        'log_dir': './logs/two_rooms/',
        'array_range': (0,10),
        'color': 'red',
    },
    {
        'name': 'Fixed Goal',
        'log_dir': './logs/two_rooms_fixed_goal/',
        'array_range': (0,10),
        'color': 'purple',
    },
    {
        'name': '50/50 Goal',
        'log_dir': './logs/two_rooms_fifty_fifty_goal/',
        'array_range': (0,10),
        'color': 'blue',
    },
    {
        'name': '50/50 TR=64',
        'log_dir': './logs/two_rooms_fifty_fifty_goal_64_train_ratio/',
        'array_range': (0,10),
        'color': 'orange',
    }
]

def main():
    create_output_directory()
    colors = [agent['color'] for agent in agents]
    total_reward_bar_data = [{'name':agent['name'], 'total_reward':None, 'err':None} for agent in agents]
    total_reward_line_data = [{'name':agent['name'], 'steps':None, 'total_reward':None} for agent in agents]
    avg_reward_data = [{'name':agent['name'], 'steps':None, 'average_reward_mean':None, 'average_reward_derivative':None} for agent in agents]
    episode_lengths_data = [{'name':agent['name'], 'episode_lengths':None, 'err':None} for agent in agents]
    for idx,agent_dict in enumerate(agents):
        tr, conf = get_average_total_reward(agent_dict)
        total_reward_bar_data[idx]['total_reward'] = tr; total_reward_bar_data[idx]['err'] = conf
        x, tr_cumsum = get_average_total_reward_cumsum(agent_dict)
        total_reward_line_data[idx]['steps'] = x; total_reward_line_data[idx]['total_reward'] = tr_cumsum
        _, avg_m = calc_average_rewards(x, tr_cumsum, type_of_average='cumsum')
        avg_reward_data[idx]['steps'] = x; avg_reward_data[idx]['average_reward_mean'] = avg_m
        _, avg_d = calc_average_rewards(x, tr_cumsum, type_of_average='derivative')
        avg_reward_data[idx]['average_reward_derivative'] = avg_d
        ep_lengths, conf = get_episode_lengths(agent_dict)
        episode_lengths_data[idx]['episode_lengths'] = ep_lengths; episode_lengths_data[idx]['err'] = conf
    plot_total_reward_bar_graph(total_reward_bar_data, output_path='total_reward.png', colors=colors)
    plot_total_reward_line_graph(total_reward_line_data, output_path='cumulative_reward.png', colors=colors)
    plot_average_reward_graph([{'name':d['name'],'steps':d['steps'],'average_reward':d['average_reward_mean']} for d in avg_reward_data], output_path='avgreward_mean.png', colors=colors)
    plot_average_reward_graph([{'name':d['name'],'steps':d['steps'],'average_reward':d['average_reward_derivative']} for d in avg_reward_data], output_path='avgreward_derivative.png', colors=colors)
    plot_episode_lengths(episode_lengths_data, output_path='episode_lengths.png', colors=colors)


if __name__ == '__main__':
    main()
