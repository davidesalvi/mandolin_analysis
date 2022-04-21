# IMPORTS
#from __future__ import division
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

import pandas as pd
import numpy as np
from numpy import random
import latex
import seaborn as sns
from collections import Counter
from scipy.stats import pearsonr
import statsmodels
import squarify # pip install squarify
import matplotlib.cm as cm


def survey_part1(csv_path):

    data_1 = pd.read_csv(csv_path)

    data_1.drop(data_1.columns[:2], axis=1, inplace=True)
    data_1.drop(data_1.columns[22:], axis=1, inplace=True)

    adj = data_1.iloc[:, 0]
    adj_list = adj.to_list()
    ad_list = [l.split(',') for l in ','.join(adj_list).split('|')][0]

    adj_list = []
    for elem in ad_list:
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

    f, ax = plt.subplots(figsize=fsize, dpi=dpi_param)
    sns.set_theme(style="whitegrid")
    clrs = [blue_color if (x < num_occurrences[0][4]) else green_color for x in num_occurrences[0]]
    b = sns.barplot(x=0, y="index", data=num_occurrences,
                    label="Total", palette=clrs)
    b.set_xticklabels(y_ticks, size=fontsize)
    b.set_yticklabels(np.array(num_occurrences.iloc[:, 0]), size=fontsize - 4)
    plt.xlabel(r'$\mathrm{Number\ of\ votes}$', size=fontsize)
    plt.ylabel(None)
    # plt.title(r'$\mathrm{Finding\ of\ the\ evaluation\ parameters}$', size=fontsize + 2, pad=15)

    plt.savefig('../figures/final_figures/adjective_couples.pdf', bbox_inches='tight')
    plt.show()


def prepare_data(csv_path, std_min=1):

    data = pd.read_csv(csv_path)

    data = data.drop(['Timestamp', 'Email Address'], axis=1)
    for idx, language in data.iloc[:, 0].items():
        if language == 'Italiano':
            data.iloc[idx, 63:125] = np.array(data.iloc[idx, 1:63])
        else:
            data.iloc[idx, 1:63] = np.array(data.iloc[idx, 63:125])

    data.drop(data.columns[1:63], axis=1, inplace=True)

    data.loc[data[data.columns[2]] == "Conservatory graduate / Professional player", data.columns[
        2]] = "Graduate/Professionist"
    data.loc[data[data.columns[2]] == 'Self-taught / amateur', data.columns[2]] = "Amateur"
    data.loc[data[data.columns[2]] == 'I took a few lessons in the past', data.columns[2]] = "Few\ lessons"
    data.loc[data[data.columns[2]] == 'Currently student', data.columns[2]] = "Student"

    data.loc[data[data.columns[0]] == 'Italiano', data.columns[0]] = "Italian"
    data.loc[data[data.columns[1]] == 'Sì', data.columns[1]] = 'Yes'
    data.loc[data[data.columns[2]] == 'Autodidatta / dilettante', data.columns[2]] = "Amateur"
    data.loc[data[data.columns[2]] == 'Ho preso qualche lezione in passato', data.columns[2]] = "Few\ lessons"
    data.loc[data[data.columns[2]] == 'Attualmente studente', data.columns[2]] = "Student"
    data.loc[data[data.columns[2]] == 'Diplomato in conservatorio / Professionista', data.columns[
        2]] = "Graduate/Professionist"
    data.loc[data[data.columns[3]] == 'Sì', data.columns[3]] = 'Yes'
    columns = range(44, 62)
    for col in columns:
        data.loc[data[data.columns[col]] == 'Mandolino 1', data.columns[col]] = 'Mandolin 1'
        data.loc[data[data.columns[col]] == 'Mandolino 2', data.columns[col]] = 'Mandolin 2'
        data.loc[data[data.columns[col]] == 'Non saprei', data.columns[col]] = 'I don\'t know'
        data.loc[data[data.columns[col]] == 'Non saprei / Sono molto simili', data.columns[col]] = 'I don\'t know'
    data = data.rename(columns={data.columns[0]: 'Language'})

    # Discard values with std < std_min
    if std_min:
        data_part = data.iloc[:, 4:43]
        # mean_val = np.mean(data2, axis=1)
        std_val = np.std(data_part, axis=1)
        data = data.loc[std_val > std_min, :]

    # Order values according to the votes of survey 1
    data.iloc[:,4:44] = 7 - data.iloc[:,4:44]
    data.iloc[:,[8, 16, 24, 32, 40]] = 7 - data.iloc[:,[8, 16, 24, 32, 40]]
    return data


def squareplot_background(data):

    sizes = []
    musicians = list(data.iloc[:, 1].value_counts())
    sizes.append(musicians[1] / sum(musicians))  # first size is non musicians
    study_path = list(data.iloc[:, 2].value_counts())
    for s in study_path:
        sizes.append((1 - sizes[0]) * s / sum(study_path))

    # colors = ["white"] + sns.color_palette("Greens_r", n_colors=5)[1:5]
    colors = ["grey"] + sns.color_palette("YlOrBr", n_colors=5)[1:5]

    fig, ax = plt.subplots(1, figsize=(6,5), dpi=dpi_param)
    labels = ["Non\ musician"] + list(data.iloc[:, 2].value_counts().index.values)
    labels = [r'$\mathrm{' + l + '}$' for l in labels]

    for i in range(len(labels)):
        labels[i] = labels[i] + "\n$(" + str(round(sizes[i] * 100, 2)) + "\%)$"
    squarify.plot(sizes=sizes, label=labels, alpha=0.55, color=colors,
                  bar_kwargs=dict(linewidth=2, edgecolor="#003319"),
                  text_kwargs={'fontsize': fontsize - 6, 'wrap': True})
    # plt.plot([0,100*(sizes[0]+sizes[1]),100*(sizes[0]+sizes[1])],[240*sizes[0],240*sizes[0],0],color='black', lw=4)
    # plt.title(r'$\mathrm{Audience\ Musical\ Background}$', fontsize=fontsize, pad=15)
    plt.yticks([], [])
    plt.xticks([], [])
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('2')

    plt.savefig('../figures/final_figures/background_square.pdf', bbox_inches='tight')

    plt.show()


def plot_single_instrument_boxplot(data, start, end, title, show_swarmData=True):
    font_size = 14

    plt.figure(figsize=(5, 4), dpi=dpi_param)

    labels_top = [r'$\mathrm{Clear}$', r'$\mathrm{Warm}$', r'$\mathrm{Brilliant}$', r'$\mathrm{Round}$',
                     r'$\mathrm{Homog.}$', r'$\mathrm{Open}$']
    labels_bottom = [r'$\mathrm{Dull}$', r'$\mathrm{Cold}$', r'$\mathrm{Opaque}$', r'$\mathrm{Sharp}$',
                  r'$\mathrm{Ringing}$', r'$\mathrm{Closed}$']
    labels_comparison = [r'$\mathrm{WWDF2}$', r'$\mathrm{WWDF1}$', r'$\mathrm{WWDF3}$', r'$\mathrm{CA-STD}$',
                         r'$\mathrm{Pandini}$']

    # colors = ["blue","orange","green","red","purple","brown"]
    my_pal = sns.color_palette()  # default palette

    # sns.set_theme(style="whitegrid")
    if end:
        ax = sns.boxplot(data=data.iloc[:, start:end], palette=my_pal)
        if show_swarmData:
            swarmData = data.iloc[:, start:end] + (
                        random.rand(data.iloc[:, start:end].shape[0], data.iloc[:, start:end].shape[1]) / 4 - 0.125)
            sns.swarmplot(data=swarmData, linewidth=1, palette=my_pal)

        # ax = sns.violinplot(data=data.iloc[:, start:end])
        ax.set_xticks(np.arange(len(labels_bottom)))
        ax.set_xticklabels(labels_bottom, fontdict={'fontsize': font_size}, rotation=0)
        ax.set_yticklabels([r'$0$', r'$1$', r'$2$', r'$3$', r'$4$', r'$5$', r'$6$'], fontdict={'fontsize': font_size})

        secax = ax.secondary_xaxis('top', functions=None)
        secax.set_xticks(np.arange(len(labels_top)))
        secax.set_xticklabels(labels_top, fontdict={'fontsize': font_size}, rotation=0)
    else:
        ax = sns.boxplot(data=data.iloc[:, start], palette=my_pal)
        # ax = sns.violinplot(data=data.iloc[:, start])
        ax.set_xticks(np.arange(len(labels_comparison)))
        ax.set_xticklabels(labels_comparison, fontdict={'fontsize': font_size})
        ax.set_yticklabels([r'$0$', r'$1$', r'$2$', r'$3$', r'$4$', r'$5$', r'$6$'], fontdict={'fontsize': font_size})
    ax.grid()
    ax.set_title('$\mathrm{' + title.replace(" ", "\;") + '}$', fontsize=font_size + 5, y=1.13)

    plt.savefig(f'../figures/final_figures/boxplot_{title[:2]}.pdf', bbox_inches='tight')

    plt.show()


def plot_feature_matrix(data):

    # order according to the the paper
    mand3 = np.mean(data.iloc[:, 4:10], axis=0).to_list()
    mand2 = np.mean(data.iloc[:, 12:18], axis=0).to_list()
    mand4 = np.mean(data.iloc[:, 20:26], axis=0).to_list()
    mand5 = np.mean(data.iloc[:, 28:34], axis=0).to_list()
    mand1 = np.mean(data.iloc[:, 36:42], axis=0).to_list()
    # mand_all = np.array([mand1, mand2, mand3, mand4, mand5])

    # mand_all = np.vstack([mand_all, mand_all[:1, :]])
    # mand_all = np.hstack([mand_all, mand_all[:, :1]])

    adjectives = [r'$\mathrm{Clear}$', r'$\mathrm{Cold}$', r'$\mathrm{Opaque}$', r'$\mathrm{Sharp}$',
                  r'$\mathrm{Homog.}$', r'$\mathrm{Closed}$']
    mandolins = [r'$\mathrm{M1}$', r'$\mathrm{M2}$', r'$\mathrm{M3}$', r'$\mathrm{M4}$', r'$\mathrm{M5}$']

    features_grid = np.array([mand1, mand2, mand3, mand4, mand5])

    fig, ax = plt.subplots(1, figsize=fsize, dpi=dpi_param)
    sns.set_style("white", {'axes.grid': False})

    plt.imshow(features_grid, cmap='RdBu')
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathrm{Average\ of\ the\ responses}$', rotation=270, size=22, labelpad=40)
    cbar.ax.tick_params(labelsize=20)

    # plt.title(r'$\mathrm{2D\ Visualization\ of\ the\ answers}$', fontsize=fontsize, pad=15)
    plt.xticks(range(6), adjectives, size=22, rotation=30)
    plt.yticks(range(5), mandolins, size=22)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('3')

    plt.savefig('../figures/final_figures/2D_features.pdf', bbox_inches='tight')

    plt.show()


def stacked_bar_comparison(data, start, end, instrument1, instrument2):
    # y_pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
    y_pos = [i for i in range(1, 24) if i % 4 != 0]
    y_pos = [2, 6, 10, 14, 18, 22]

    performance = pd.Series(dtype='int64')
    for idx in range(start, end):
        responses = data.iloc[:, idx].value_counts()
        responses = responses.reindex(['Mandolin 1', 'Mandolin 2', 'I don\'t know'])
        performance = pd.concat([performance, responses])

    # color_vec = ('#2e5cb7', '#c63310', '#e68a00')
    # colors = sns.color_palette("hls",n_colors=5)[2:5]
    # color_vec = tuple(colors)
    colors = sns.color_palette("colorblind")
    color_vec = (colors[0], colors[3], colors[1])
    # mandolins = [instrument1, instrument2, "Don\'t know"]
    mandolins_legend = ['$\mathrm{' + instrument1 + '}$', '$\mathrm{' + instrument2 + '}$', "$\mathrm{Don't\;know}$"]
    cmap = dict(zip(mandolins_legend, color_vec))
    patches = [Patch(color=v, label=k) for k, v in cmap.items()]

    font_size = 18
    plt.figure(figsize=(12, 3.5), dpi=120)
    plt.grid(b=True, axis='y', zorder=2)

    # plt.bar(y_pos, performance, align='center', color=(color_vec * 6), width=.95, zorder=2)
    total = np.add(np.add(performance[::3].array, performance[1::3].array), performance[2::3].array)

    plt.bar(y_pos, (performance[::3] / total) * 100, align='center', color=colors[0], width=.95, zorder=2)
    plt.bar(y_pos, (performance[1::3] / total) * 100, bottom=(performance[::3].array / total) * 100, align='center',
            color=colors[3], width=.95, zorder=2)
    plt.bar(y_pos, (performance[2::3] / total) * 100,
            bottom=np.add((performance[1::3].array / total) * 100, (performance[::3].array / total) * 100),
            align='center', color=colors[1], width=.95, zorder=2)

    # plt.bar(y_pos, performance[::3], align='center', color=colors[0], width=.95, zorder=2)
    # plt.bar(y_pos, performance[1::3], bottom=performance[::3], align='center', color=colors[1], width=.95, zorder=2)
    # plt.bar(y_pos, performance[2::3], bottom = np.add(performance[1::3].array,performance[::3].array), align='center', color=colors[2], width=.95, zorder=2)

    plt.xticks([2, 6, 10, 14, 18, 22], [r'$\mathrm{Brilliant}$', r'$\mathrm{Round}$', r'$\mathrm{Warm}$',
                                        r'$\mathrm{Soft}$', r'$\mathrm{Sustain}$', r'$\mathrm{Overall}$'],
               rotation=0, fontsize=font_size + 2)

    plt.ylim(0, 100)
    plt.tick_params(axis='y', labelsize=font_size - 5)
    plt.title('$\mathrm{' + instrument1 + ' - ' + instrument2 + '}$', fontsize=font_size + 4, pad=20)
    plt.legend(mandolins_legend, handles=patches, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=font_size)

    plt.savefig(f'../figures/final_figures/comparison_{instrument1}_{instrument2}.pdf', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':

    # Define parameters
    fsize = (9, 6)
    dpi_param = 120
    fontsize = 20
    red_color = '#a45f5a'
    blue_color = '#5ea7c1'
    green_color = '#5ec164'

    # # SURVEY PART 1
    # csv_path_part1 = '../data/part1_15.csv'
    # survey_part1(csv_path_part1)

    # SURVEY PART 2
    csv_path_part2 = '../data/part2_111.csv'
    data = prepare_data(csv_path_part2, std_min=1)
    # squareplot_background(data)

    # # single instrument box-plots
    # # plot_single_instrument_boxplot(data, 4, 10, 'M1 : WWDF2')
    # # plot_single_instrument_boxplot(data, 12, 18, 'M2 : WWDF1')
    # # plot_single_instrument_boxplot(data, 20, 26, 'M3 : WWDF3')
    # # plot_single_instrument_boxplot(data, 28, 34, 'M4 : CA-STD')
    # # plot_single_instrument_boxplot(data, 36, 42, 'M5 : Pandini')

    plot_single_instrument_boxplot(data, 4, 10, 'M3')# : WWDF2')
    plot_single_instrument_boxplot(data, 12, 18, 'M2')# : WWDF1')
    plot_single_instrument_boxplot(data, 20, 26, 'M4')# : WWDF3')
    plot_single_instrument_boxplot(data, 28, 34, 'M5')# : CA-STD')
    plot_single_instrument_boxplot(data, 36, 42, 'M1')# : Pandini')

    # 2D matrix with features
    plot_feature_matrix(data)

    # mandolin comparison
    stacked_bar_comparison(data, 44, 50, 'M1', 'M4')
    stacked_bar_comparison(data, 50, 56, 'M5', 'M1')
    stacked_bar_comparison(data, 56, 62, 'M4', 'M5')



