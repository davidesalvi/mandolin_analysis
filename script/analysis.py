import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

# make plots using LaTeX font
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

def prepare_data(csv_path, std_min=None):

    data = pd.read_csv(csv_path)

    # Remove useless columns
    data = data.drop(['Timestamp', 'Email Address'], axis=1)

    # Copy all italian responses in the english columns and vice versa
    for idx, language in data.iloc[:, 0].items():
        if language == 'Italiano':
            data.iloc[idx, 63:125] = np.array(data.iloc[idx, 1:63])
        else:
            data.iloc[idx, 1:63] = np.array(data.iloc[idx, 63:125])

    # Remove all the italian columns and keep only the english ones
    data.drop(data.columns[1:63], axis=1, inplace=True)

    # Make all the answers in english

    columns = [0]
    for col in columns:
        for idx, answer in data.iloc[:, col].items():
            if answer == 'Italiano':
                data.iloc[idx, col] = 'Italian'

    columns = [2]
    for col in columns:
        for idx, answer in data.iloc[:, col].items():
            if answer == 'Autodidatta / dilettante':
                data.iloc[idx, col] = 'Self-taught / amateur'
            elif answer == 'Ho preso qualche lezione in passato':
                data.iloc[idx, col] = 'I took a few lessons in the past'
            elif answer == 'Attualmente studente':
                data.iloc[idx, col] = 'Currently student'
            elif answer == 'Diplomato in conservatorio / Professionista':
                data.iloc[idx, col] = 'Conservatory graduate / Professional player'

    columns = [1, 3]
    for col in columns:
        for idx, answer in data.iloc[:, col].items():
            if answer == 'Sì':
                data.iloc[idx, col] = 'Yes'

    columns = range(44, 62)
    for col in columns:
        for idx, answer in data.iloc[:, col].items():
            if answer == 'Mandolino 1':
                data.iloc[idx, col] = 'Mandolin 1'
            elif answer == 'Mandolino 2':
                data.iloc[idx, col] = 'Mandolin 2'
            elif answer == 'Non saprei':
                data.iloc[idx, col] = 'I don\'t know'
            elif answer == 'Non saprei / Sono molto simili':
                # data.iloc[idx, col] = 'I don\'t know / They are very similar'
                data.iloc[idx, col] = 'I don\'t know'


    data = data.rename(columns={data.columns[0]: 'Language'})

    # data_mod = data[data.iloc[:,2] == 'Conservatory graduate / Professional player']
    # data_mod = data[data.iloc[:,3] == 'Yes']
    # data_std = np.sum(np.std(data_mod.iloc[:,4:43], axis=0))

    # Delete all the interviewed where std < 1 (gave the same answer to all everything)
    if std_min:
        data_part = data.iloc[:,4:43]
        # mean_val = np.mean(data2, axis=1)
        std_val = np.std(data_part, axis=1)
    
        data = data.loc[std_val > std_min,:]

    return data


def audience_analysis(data):

    plt.title('Survey language')
    plt.pie(data.iloc[:, 0].value_counts(), autopct='%.1f\%%')
    plt.legend(data.iloc[:, 0].value_counts().index.values)
    plt.show()

    # data = data[data['Language'] == 'Italian']

    plt.title('Musician?')
    plt.pie(data.iloc[:, 1].value_counts(), autopct='%.1f\%%')
    plt.legend(data.iloc[:, 1].value_counts().index.values)
    plt.show()

    plt.title('Music study path')
    plt.pie(data.iloc[:, 2].value_counts(), autopct='%.1f\%%')
    plt.legend(data.iloc[:, 2].value_counts().index.values)
    plt.show()

    plt.title('Plucked instruments player')
    plt.pie(data.iloc[:, 3].value_counts(), autopct='%.1f\%%')
    plt.legend(data.iloc[:, 3].value_counts().index.values)
    plt.show()


def plot_single_instrument_boxplot(data, start, end, title, save_name):

    font_size = 12

    labels_top = [r'$\mathrm{Dull}$',r'$\mathrm{Cold}$',r'$\mathrm{Opaque}$',r'$\mathrm{Sharp}$',
                  r'$\mathrm{Homog.}$',r'$\mathrm{Closed}$',r'$\mathrm{Sustain}$']
    labels_bottom = [r'$\mathrm{Clear}$',r'$\mathrm{Warm}$',r'$\mathrm{Brilliant}$',r'$\mathrm{Round}$',
                     r'$\mathrm{Ringing}$',r'$\mathrm{Open}$',r'$\mathrm{Sustain}$']
    labels_comparison = [r'$\mathrm{WWDF2}$',r'$\mathrm{WWDF1}$',r'$\mathrm{WWDF3}$',r'$\mathrm{CA-STD}$',
                     r'$\mathrm{Pandini}$']

    sns.set_theme(style="whitegrid")
    if end:
        # ax = sns.boxplot(data=data.iloc[:, start:end])
        ax = sns.violinplot(data=data.iloc[:, start:end])
        ax.set_xticks(np.arange(len(labels_bottom)))
        ax.set_xticklabels(labels_bottom, fontdict={'fontsize': font_size})

        secax = ax.secondary_xaxis('top', functions=None)
        secax.set_xticks(np.arange(len(labels_top)))
        secax.set_xticklabels(labels_top, fontdict={'fontsize': font_size})
    else:
        # ax = sns.boxplot(data=data.iloc[:, start])
        ax = sns.violinplot(data=data.iloc[:, start])
        ax.set_xticks(np.arange(len(labels_comparison)))
        ax.set_xticklabels(labels_comparison, fontdict={'fontsize': font_size})

    plt.title('$\mathrm{' + title.replace(" ", "\;") + '}$', fontsize= font_size+2, y=1.13)

    # plt.savefig(f'figures/{save_name}_violin.pdf', bbox_inches='tight')
    plt.show()


def plot_single_instrument_errorbar(data, start, end, title, save_name):

    font_size = 12

    labels_top = [r'$\mathrm{Dull}$',r'$\mathrm{Cold}$',r'$\mathrm{Opaque}$',r'$\mathrm{Sharp}$',
                  r'$\mathrm{Homog.}$',r'$\mathrm{Closed}$',r'$\mathrm{Sustain}$']
    labels_bottom = [r'$\mathrm{Clear}$',r'$\mathrm{Warm}$',r'$\mathrm{Brilliant}$',r'$\mathrm{Round}$',
                     r'$\mathrm{Ringing}$',r'$\mathrm{Open}$',r'$\mathrm{Sustain}$']
    labels_comparison = [r'$\mathrm{WWDF2}$',r'$\mathrm{WWDF1}$',r'$\mathrm{WWDF3}$',r'$\mathrm{CA-STD}$',
                     r'$\mathrm{Pandini}$']

    plt.figure(figsize=(6, 4.5), dpi=120)
    sns.set_theme(style="whitegrid")
    if end:
        ax = sns.pointplot(data=data.iloc[:, start:end], join=False, ci='sd')
        ax.set_xticks(np.arange(len(labels_bottom)))
        ax.set_xticklabels(labels_bottom, fontdict={'fontsize': font_size})
        plt.hlines(3.5, -0.5, 6.5, colors='k', linestyles='dashed')
        plt.xlim(-0.5,6.5)
        plt.ylim(0.8,6.2)

        secax = ax.secondary_xaxis('top', functions=None)
        secax.set_xticks(np.arange(len(labels_top)))
        secax.set_xticklabels(labels_top, fontdict={'fontsize': font_size})
    else:
        # ax = sns.pointplot(data=data.iloc[:, start])
        ax = sns.violinplot(data=data.iloc[:, start])
        ax.set_xticks(np.arange(len(labels_comparison)))
        ax.set_xticklabels(labels_comparison, fontdict={'fontsize': font_size})

    plt.title('$\mathrm{' + title.replace(" ", "\;") + '}$', fontsize= font_size+2, y=1.13)

    # plt.savefig(f'figures/{save_name}_violin.pdf', bbox_inches='tight')
    plt.show()


    # from sklearn.feature_selection import f_regression
    # from scipy import stats
    #
    # X_val = np.array(data.iloc[:, start:end])
    # y_val = np.ones(X_val.shape[0])*3.5
    #
    # slope, intercept, r_value, p_value, std_err = stats.linregress(X_val[:,0], y_val)
    # print(p_value)
    #
    # eps = 1e-6
    #
    #
    # f_statistics, p_values = f_regression(X_val, y_val)



def single_analysis(data):


    plot_single_instrument_boxplot(data, 4, 11, 'Mandolin 1 - WWDF2', 'wwdf2')
    plot_single_instrument_boxplot(data, 12, 19, 'Mandolin 2 - WWDF1', 'wwdf1')
    plot_single_instrument_boxplot(data, 20, 27, 'Mandolin 3 - WWDF3', 'wwdf3')
    plot_single_instrument_boxplot(data, 28, 35, 'Mandolin 4 - CA-STD', 'castd')
    plot_single_instrument_boxplot(data, 36, 43, 'Mandolin 5 - Pandini', 'pandini')
    plot_single_instrument_boxplot(data, [11, 19, 27, 35, 43], None, 'Overall Comparison', 'overall_comp')

    plot_single_instrument_errorbar(data, 4, 11, 'Mandolin 1 - WWDF2', 'wwdf2')
    plot_single_instrument_errorbar(data, 12, 19, 'Mandolin 2 - WWDF1', 'wwdf1')
    plot_single_instrument_errorbar(data, 20, 27, 'Mandolin 3 - WWDF3', 'wwdf3')
    plot_single_instrument_errorbar(data, 28, 35, 'Mandolin 4 - CA-STD', 'castd')
    plot_single_instrument_errorbar(data, 36, 43, 'Mandolin 5 - Pandini', 'pandini')
    plot_single_instrument_errorbar(data, [11, 19, 27, 35, 43], None, 'Overall Comparison', 'overall_comp')


def plot_comparison(data, start, end, instrument1, instrument2, save_name):

    y_pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]

    performance = pd.Series(dtype='int64')
    for idx in range(start, end):
        responses = data.iloc[:, idx].value_counts()
        responses = responses.reindex(['Mandolin 1', 'Mandolin 2', 'I don\'t know'])
        performance = pd.concat([performance, responses])

    color_vec = ('#2e5cb7', '#c63310', '#e68a00')

    # mandolins = [instrument1, instrument2, "Don\'t know"]
    mandolins_legend = ['$\mathrm{' + instrument1 + '}$', '$\mathrm{' + instrument2 + '}$', "$\mathrm{Don't\;know}$"]
    cmap = dict(zip(mandolins_legend, color_vec))
    patches = [Patch(color=v, label=k) for k, v in cmap.items()]

    font_size = 18
    plt.figure(figsize=(12, 3.5), dpi=120)
    plt.grid(b=True, which='major', axis='y', zorder=0)
    plt.bar(y_pos, performance, align='center', color=(color_vec * 6), width=.95, zorder=2)
    # plt.xticks([2, 6, 10, 14, 18, 22], ['Brilliant', 'Round', 'Warm', 'Soft', 'Sustain', 'Overall \n preference'],
    plt.xticks([2, 6, 10, 14, 18, 22], [r'$\mathrm{Brilliant}$', r'$\mathrm{Round}$', r'$\mathrm{Warm}$',
                r'$\mathrm{Soft}$', r'$\mathrm{Sustain}$', r'$\mathrm{Overall}\\ \mathrm{performance}$'],
               rotation=0, fontsize=font_size+2)

    plt.ylim(0, 105)
    plt.tick_params(axis='y', labelsize=font_size-5)
    plt.title('$\mathrm{' + instrument1 + ' - ' + instrument2 + '}$', fontsize=font_size + 4)
    plt.legend(mandolins_legend, handles=patches, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=font_size)

    # plt.savefig(f'figures/{save_name}.pdf', bbox_inches='tight')
    plt.show()

def comparison_analysis(data):

    # data = data[data['Language'] == 'Italian']

    plot_comparison(data, 44, 50, 'Pandini', 'WWDF3', 'comparison1')
    plot_comparison(data, 50, 56, 'CA-STD', 'Pandini', 'comparison2')
    plot_comparison(data, 56, 62, 'WWDF3', 'CA-STD', 'comparison3')


    print()

if __name__ == '__main__':

    csv_path = 'data/part2_109.csv'

    data = prepare_data(csv_path, std_min=1)

    audience_analysis(data)
    single_analysis(data)
    comparison_analysis(data)

    print()