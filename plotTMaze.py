import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import stats


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


def get_average_total_reward(agent_dict):
    total_rewards = []
    for seed in range(*agent_dict['array_range']):
        total_reward = 0
        with open(f"{agent_dict['log_dir']}/{seed}/metrics.jsonl", 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            if 'episode/score' in result:
                total_reward += result['episode/score']
        total_rewards.append(total_reward)
    m, _, conf = mean_confidence_interval(total_rewards)
    return m, conf

def plot_total_reward_graph(total_rewards, output_path='barplot.png', colors=None):
    names = [d['name'] for d in total_rewards]
    heights = [d['total_reward'] for d in total_rewards]
    conf_intervals = [d['err'] for d in total_rewards]
    print(conf_intervals)
    plt.bar(names, heights, yerr=conf_intervals, capsize=10, color=colors, edgecolor='black')
    plt.ylabel('Total Reward')
    plt.savefig(output_path)
    plt.clf()


agents = [
    {
        'name': 'Uniform Parameterized Fifo Dreamer',
        'log_dir': './logs/tmaze/',
        'array_range': (0,9),
        'color': 'red',
    },
    {
        'name': 'With 0 padded obs',
        'log_dir': './logs/tmaze_zero_pad/',
        'array_range': (0,9),
        'color': 'gray',
    },
]

def main():
    colors = [agent['color'] for agent in agents]
    total_reward_data = [{'name':agent['name'], 'total_reward':None, 'err':None} for agent in agents]
    for idx,agent_dict in enumerate(agents):
        tr, conf = get_average_total_reward(agent_dict)
        total_reward_data[idx]['total_reward'] = tr; total_reward_data[idx]['err'] = conf
    plot_total_reward_graph(total_reward_data, output_path='total_reward.png', colors=colors)


if __name__ == '__main__':
    main()
