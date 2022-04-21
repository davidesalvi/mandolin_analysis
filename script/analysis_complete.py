import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

# make plots using LaTeX font
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

csv_path = '../data/part2_109.csv'
data = pd.read_csv(csv_path)

# Prepare data
data = data.drop(['Timestamp', 'Email Address'], axis=1)
for idx, language in data.iloc[:, 0].items():
    if language == 'Italiano':
        data.iloc[idx, 63:125] = np.array(data.iloc[idx, 1:63])
    else:
        data.iloc[idx, 1:63] = np.array(data.iloc[idx, 63:125])
data.drop(data.columns[1:63], axis=1, inplace=True)

# Shortify answers
data.loc[data[data.columns[2]] == "Conservatory graduate / Professional player", data.columns[2]] = "Graduate/Professionist"
data.loc[data[data.columns[2]] == 'Self-taught / amateur',data.columns[2]] = "Amateur"
data.loc[data[data.columns[2]] == 'I took a few lessons in the past',data.columns[2]] = "Few lessons"

# Make all answers in English
data.loc[data[data.columns[0]] == 'Italiano', data.columns[0]] = "Italian"
data.loc[data[data.columns[1]] == 'Sì', data.columns[1]] = 'Yes'
data.loc[data[data.columns[2]] == 'Autodidatta / dilettante', data.columns[2]] = "Amateur"
data.loc[data[data.columns[2]] == 'Ho preso qualche lezione in passato', data.columns[2]]  = "Few lessons"
data.loc[data[data.columns[2]] == 'Attualmente studente', data.columns[2]]  = "Currently student"
data.loc[data[data.columns[2]] == 'Diplomato in conservatorio / Professionista', data.columns[2]]  = "Graduate/Professionist"
data.loc[data[data.columns[3]] == 'Sì', data.columns[3]] = 'Yes'
columns = range(44, 62)
for col in columns:
    data.loc[data[data.columns[col]] == 'Mandolino 1', data.columns[col]] = 'Mandolin 1'
    data.loc[data[data.columns[col]] == 'Mandolino 2', data.columns[col]] = 'Mandolin 2'
    data.loc[data[data.columns[col]] == 'Non saprei', data.columns[col]] = 'I don\'t know'
    data.loc[data[data.columns[col]] == 'Non saprei / Sono molto simili', data.columns[col]] = 'I don\'t know'
data = data.rename(columns={data.columns[0]: 'Language'})

# AUDIENCE ANALYSIS - PIE PLOT
fig, axes = plt.subplots(2,2,figsize=(12,12))
# titles = ['Survey Language','Musician?','Music Study Path','Plucked Instruments player']
titles = [r'$\mathrm{Survey\;Language}$',r'$\mathrm{Musician}$',r'$\mathrm{Music\;Study\;Path}$',r'$\mathrm{Plucked\;Instruments\;Player}$']
for i in range(4):
    labels = list(data.iloc[:, i].value_counts().index.values)
    axes[int(i/2)][i%2].pie(data.iloc[:, i].value_counts(), autopct='%.1f\%%',labels = labels)
    axes[int(i/2)][i%2].set_title(titles[i])
    #axes[int(i/2)][i%2].legend(data.iloc[:, i].value_counts().index.values)
#axes[1,0].legend(data.iloc[:, 2].value_counts().index.values,bbox_to_anchor=(1.5,0.2))
plt.subplots_adjust(hspace=0,wspace=0.6)
# plt.plot([1.3,1.3], [1.1,-1.4], color='black', lw=1, transform=axes[0][0].transAxes, clip_on=False) #vertical line
# plt.plot([-0.1,2.6],[-0.1,-0.1], color='black', lw=1,transform=axes[0][0].transAxes, clip_on=False) #orizontal line
plt.show()

# AUDIENCE ANALYSIS - SQUARE PLOT
import squarify

sizes = []
musicians = list(data.iloc[:, 1].value_counts())
sizes.append(musicians[1] / sum(musicians)) #first size is non musicians
study_path = list(data.iloc[:, 2].value_counts())
for s in study_path:
    sizes.append( (1-sizes[0]) * s/sum(study_path))
print(sizes)
# colors = ["grey","red","orange","yellow","white"]
norm = mpl.colors.Normalize(vmin=min(sizes), vmax=max(sizes))
colors = [mpl.cm.YlOrBr(norm(value)) for value in sizes]
colors[0] = 'grey'
# labels = [r"$\mathrm{Non\;musician}$"] + list(data.iloc[:, 2].value_counts().index.values)
labels = [r"$\mathrm{Non\;musician}$", r"$\mathrm{Graduated/Professionist}$", r"$\mathrm{Amateur\;musician}$", r"$\mathrm{Took\;few\;lessons}$", r"$\mathrm{Currently\;student}$"]
for i in range(len(labels)):
    # '$\mathrm{' + legend + ' - AUC = %0.2f}$' % rocauc
    # labels[i] = labels[i][:-2]+"\n $\mathrm{%0.2f}\%$" % sizes[i]
    # labels[i] = labels[i][:-2]+"\\\;%0.2f\%%}$" % sizes[i]
    labels[i] = labels[i] + '\n' + r'$\mathrm{%0.2f\%%}$' % (sizes[i]*100)
squarify.plot(sizes=sizes, label=labels, alpha=0.7, color=colors, bar_kwargs=dict(linewidth=1.5, edgecolor="#222222"),text_kwargs={'fontsize':17, 'wrap':True})
# plt.plot([0,100*(sizes[0]+sizes[1]),100*(sizes[0]+sizes[1])],[200*sizes[0],200*sizes[0],0],color='black', lw=2)
# plt.title("Audience Music level")
plt.yticks([],[])
plt.xticks([],[])
# plt.savefig('../figures/square_plot.pdf')
plt.show()

# STD OF THE ANSWERS

std_min = 1
data_part = data.iloc[:, 4:43]
# mean_val = np.mean(data2, axis=1)
std_val = np.std(data_part, axis=1)
# if std_min:
#    data = data.loc[std_val > std_min, :]

fontsize = 16
plt.figure(figsize=(9, 4))
plt.scatter(range(len(std_val)), std_val)

plt.hlines(std_min, -5, len(std_val) + 5, colors='r', linestyles='dashed')

minor_val = [(x, y) for x, y in enumerate(std_val) if y < std_min]
discard_count = 0
if len(minor_val) > 0:
    xmin, ymin = zip(*minor_val)
    plt.scatter(xmin, ymin, marker="x", color='r', s=30)
    discard_count += 1

plt.xlim([-5, len(std_val) + 5])
plt.ylim((0.5, 2))
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel(r'$\mathrm{STD}$', fontsize=fontsize)
plt.xlabel(r'$\mathrm{Interviewed\;idx}$', fontsize=fontsize)
plt.grid()
# plt.title("STD of each response values")
plt.savefig('../figures/STD_values.pdf')
plt.show()

print(f'We discarded {len(minor_val)}/{len(std_val)} answers')
print(f'{len(std_val) - len(minor_val)} answers are remaining')

print()



