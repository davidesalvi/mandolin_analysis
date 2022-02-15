import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch
from collections import Counter

# make plots using LaTeX font
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

def prepare_data(csv_path):

    data = pd.read_csv(csv_path)

    data.drop(data.columns[:2], axis=1, inplace=True)
    data.drop(data.columns[22:], axis=1, inplace=True)

    return data


def plot_adjectives_couples(data):

    adj = data.iloc[:,0]
    adj_list = adj.to_list()
    list = [l.split(',') for l in ','.join(adj_list).split('|')][0]

    adj_list = []
    for elem in list:
        j = elem.replace(' ', '')
        adj_list.append(j)

    count = Counter(adj_list)
    num_occurrences = pd.DataFrame.from_dict(count, orient='index').reset_index()

    # Make titles in Latex font and enlarge fontsize
    num_occurrences.iloc[0, 0] = r'$\mathrm{RINGING-HOMOGENEOUS}$'
    num_occurrences.iloc[1, 0] = r'$\mathrm{OPEN-CLOSED}$'
    num_occurrences.iloc[2, 0] = r'$\mathrm{INTENSE-TENUOUS}$'
    num_occurrences.iloc[3, 0] = r'$\mathrm{SOFT-HARD}$'
    num_occurrences.iloc[4, 0] = r'$\mathrm{WARM-COLD}$'
    num_occurrences.iloc[5, 0] = r'$\mathrm{BRILLIANT-OPAQUE}$'
    num_occurrences.iloc[6, 0] = r'$\mathrm{POWERFUL-WEAK}$'
    num_occurrences.iloc[7, 0] = r'$\mathrm{ROUND-SHARP}$'
    num_occurrences.iloc[8, 0] = r'$\mathrm{NASAL-METALLIC}$'
    num_occurrences.iloc[9, 0] = r'$\mathrm{CLEAR-DULL}$'
    num_occurrences.iloc[10, 0] = r'$\mathrm{RICH-POOR}$'
    num_occurrences.iloc[11, 0] = r'$\mathrm{BALANCED-UNBALANCED}$'
    num_occurrences.iloc[12, 0] = r'$\mathrm{RICH-POOR\;IN\;HARMONICS}$'

    num_occurrences = num_occurrences.sort_values(by=[0], ascending=False)

    y_ticks = np.array([r'$\mathrm{0}$', r'$\mathrm{1}$', r'$\mathrm{2}$', r'$\mathrm{3}$', r'$\mathrm{4}$',
                        r'$\mathrm{5}$', r'$\mathrm{6}$', r'$\mathrm{7}$', r'$\mathrm{8}$', ])

    f, ax = plt.subplots(figsize=(12, 6), dpi=120)
    sns.set_theme(style="whitegrid")
    clrs = ['#4e76a5' if (x < num_occurrences[0][4]) else '#d1564e' for x in num_occurrences[0]]
    b = sns.barplot(x=0, y="index", data=num_occurrences,
                label="Total", palette=clrs)#color="b")
    b.set_xticklabels(y_ticks, size=18)
    b.set_yticklabels(np.array(num_occurrences.iloc[:,0]), size=16)
    ax.set(xlabel=None, ylabel=None)

    # plt.savefig(f'../figures/adjective_couples.pdf', bbox_inches='tight')

    plt.show()

    print()


def plot_sound_characteristics(data):

    charact = data.iloc[:, 12:]

    charact_names = ['Sound volume', 'Sound projection', 'Sustain', 'Balancing', 'Responsiveness', 'Dynamic range',
                       'Timbre', 'Tonal contrast', 'Separation', 'Sound quality']
    charact.columns = charact_names

    mean = np.mean(charact, axis=0)
    stdev = np.std(charact, axis=0)

    plt.figure(figsize=(13, 5))
    # plt.errorbar(np.arange(10), mean, stdev)
    sns.pointplot(data=charact, join=False, ci=90, capsize=.2)
    # plt.xticks(np.arange(10), charact_names)
    plt.yticks(np.arange(7))
    plt.ylim(0.5, 6.5)

    plt.show()


if __name__ == '__main__':

    csv_path = '../data/part1_15.csv'

    data = prepare_data(csv_path)
    plot_adjectives_couples(data)
    plot_sound_characteristics(data)