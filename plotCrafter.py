import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import stats
from pathlib import Path
import pandas as pd


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

def geometric_mean_confidence_interval(data, confidence=0.95):
    """
    Code obtained from the link below:
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    # from https://arxiv.org/pdf/2109.06780.pdf page 5, note 2
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.exp(np.sum(np.log(1 + a)) / n), stats.sem(a, axis=0)
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

def get_rewards_and_success_rates_for_seed(agent_dict, seed):
    x = []
    rewards = []
    success_rates = dict()
    with open(f"{agent_dict['log_dir']}/{seed}/crafter/stats.jsonl", 'r') as json_file:
        json_list = list(json_file)
    achievements = list(set(json.loads(json_list[0]).keys()) - {'length','reward'})
    success_rates = {ach:0 for ach in achievements}
    for json_str in json_list:
        line = json.loads(json_str)
        if len(x) == 0:
            x.append(line['length'])
        else:
            x.append(x[-1] + line['length'])
        rewards.append(line['reward'])
        for ach in achievements:
            if line[ach] > 0:
                success_rates[ach] += 1
    factor = 100 / len(json_list)
    for ach in achievements:
        success_rates[ach] *= factor
    return x, rewards, success_rates

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

def get_rewards_and_success_rates(agent_dict):
    print(f"Getting rewards and success rates for {agent_dict['name']} ...")
    data_triples = [get_rewards_and_success_rates_for_seed(agent_dict, seed) for seed in range(*agent_dict['array_range'])]
    print(f"Finished getting rewards and success rates for {agent_dict['name']}.")
    return data_triples

def calc_average_success_rates(data_triples):
    success_rates_per_seed = [triple[2] for triple in data_triples]
    averaged_rates = {ach:0 for ach in success_rates_per_seed[0]}
    for ach in averaged_rates:
        success_rates_for_ach = [srd[ach] for srd in success_rates_per_seed]
        m, _, conf = mean_confidence_interval(success_rates_for_ach)
        averaged_rates[ach] = m
    return averaged_rates

def plot_success_rates(success_rates, output_path='crafter_success_rates.png', colors=None):
    def achievement_to_str(achievement):
        words = achievement.split('_')[1:]
        words = [w[0].upper() + w[1:] for w in words]
        return ' '.join(words)
    agent_names = [sr['name'] for sr in success_rates]
    achievements = [ach for ach in success_rates[0]['success_rates']]
    data = []
    for ach in achievements:
        sr = [agent_dict['success_rates'][ach] for agent_dict in success_rates]
        data.append(sr)
    df = pd.DataFrame(data, columns=agent_names, index=achievements)
    df.plot.bar(color=colors, edgecolor='black')
    plt.legend()
    plt.ylabel('Success Rate (%)')
    plt.savefig(Path(OUTPUT_DIRECTORY) / output_path)
    plt.clf()

def calc_avg_score(avg_success_rates:dict):
    si = list(avg_success_rates.values())
    m, _, conf = geometric_mean_confidence_interval(si)
    return m, conf

def plot_crafter_scores(crafter_scores, output_path='crafter_scores.png', colors=None):
    print(f'Plotting crafter scores bar graph to {output_path} ...')
    names = [d['name'] for d in crafter_scores]
    heights = [d['score'] for d in crafter_scores]
    conf_intervals = [d['err'] for d in crafter_scores]
    plt.bar(names, heights, yerr=conf_intervals, capsize=10, color=colors, edgecolor='black')
    plt.ylabel('Crafter Score (%)')
    plt.savefig(Path(OUTPUT_DIRECTORY) / output_path)
    plt.clf()
    print(f"Finished plotting scrafter scores bar graph.")

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


OUTPUT_DIRECTORY = './plots/crafter_compare_seeding_issue/'
agents = [
    # {
    #     'name': 'Dreamer 1e6 (1)',
    #     'log_dir': './logs/crafter/',
    #     'array_range': (0,10),
    #     'color': 'blue',
    # },
    # {
    #     'name': 'Reservoir Dreamer 1e6 (4)',
    #     'log_dir': './logs/crafter_reservoir/',
    #     'array_range': (0,10),
    #     'color': 'orange',
    # },
    # {
    #     'name': 'Dreamer 1e5 (1)',
    #     'log_dir': './logs/crafter_fifo_replay_size_1e5/',
    #     'array_range': (0,10),
    #     'color': 'green',
    # },
    # {
    #     'name': 'Reservoir Dreamer 1e5 (4)',
    #     'log_dir': './logs/crafter_reservoir_replay_size_1e5/',
    #     'array_range': (0,10),
    #     'color': 'red',
    # },
    {
        'name': 'Dreamer Seed 0 1',
        'log_dir': './logs/crafter/',
        'array_range': (0,1),
        'color': 'blue',
    },
    {
        'name': 'Dreamer Seed 0 2',
        'log_dir': './logs/crafter_regular_seeds_01/',
        'array_range': (0,1),
        'color': 'purple',
    },
    {
        'name': 'Dreamer Seed 1 1',
        'log_dir': './logs/crafter/',
        'array_range': (1,2),
        'color': 'lime',
    },
    {
        'name': 'Dreamer Seed 1 2',
        'log_dir': './logs/crafter_regular_seeds_01/',
        'array_range': (1,2),
        'color': 'green',
    },
    # {
    #     'name': 'Reservoir Dreamer 1e6 (4)',
    #     'log_dir': './logs/crafter_reservoir/',
    #     'array_range': (0,10),
    #     'color': 'orange',
    # },
]

def main():
    create_output_directory()
    colors = [agent['color'] for agent in agents]
    total_reward_bar_data = [{'name':agent['name'], 'total_reward':None, 'err':None} for agent in agents]
    total_reward_line_data = [{'name':agent['name'], 'steps':None, 'total_reward':None} for agent in agents]
    avg_reward_data = [{'name':agent['name'], 'steps':None, 'average_reward_mean':None, 'average_reward_derivative':None} for agent in agents]
    episode_lengths_data = [{'name':agent['name'], 'episode_lengths':None, 'err':None} for agent in agents]
    success_rates_data = [{'name':agent['name'], 'success_rates':None} for agent in agents]
    scores_data = [{'name':agent['name'], 'score':None, 'err':None} for agent in agents]
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
        data_triples = get_rewards_and_success_rates(agent_dict)
        avg_success_rates = calc_average_success_rates(data_triples)
        success_rates_data[idx]['success_rates'] = avg_success_rates
        score, conf = calc_avg_score(avg_success_rates)
        scores_data[idx]['score'] = score; scores_data[idx]['err'] = conf
    plot_total_reward_bar_graph(total_reward_bar_data, output_path='total_reward.png', colors=colors)
    plot_total_reward_line_graph(total_reward_line_data, output_path='cumulative_reward.png', colors=colors)
    plot_average_reward_graph([{'name':d['name'],'steps':d['steps'],'average_reward':d['average_reward_mean']} for d in avg_reward_data], output_path='avgreward_mean.png', colors=colors)
    plot_average_reward_graph([{'name':d['name'],'steps':d['steps'],'average_reward':d['average_reward_derivative']} for d in avg_reward_data], output_path='avgreward_derivative.png', colors=colors)
    plot_episode_lengths(episode_lengths_data, output_path='episode_lengths.png', colors=colors)
    plot_success_rates(success_rates_data, colors=colors)
    plot_crafter_scores(scores_data, colors=colors)


if __name__ == '__main__':
    main()
