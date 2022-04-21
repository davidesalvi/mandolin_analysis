import pandas as pd
import numpy as np
from scipy import stats


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


def main(data):

    idx_start = [4, 12, 20, 28, 36]
    # idx_end = [10, 18, 26, 34, 42]
    labels = ['M3', 'M2', 'M4', 'M5', 'M1']
    features_top = ['clear', 'warm', 'brilliant', 'round', 'homogeneous', 'open']
    features_bottom = ['dull', 'cold', 'opaque', 'sharp', 'ringing', 'closed']

    null_hp = 3.5

    num_weak = 0
    num_strong = 0
    num_no = 0

    for idx in range(len(labels)):

        print()
        print(f'Analyzing instrument {labels[idx]}')
        print()

        for feat_idx in range(6):

            results = np.array(data.iloc[:, feat_idx+idx_start[idx]])

            # t_test = (np.mean(results) - null_hp) / (np.std(results-null_hp) / np.sqrt(len(results)))
            t_test, p_value = stats.ttest_ind(results, results*0+null_hp)

            if np.abs(p_value*2) < 0.01:
                evidence = 'VERY STRONG EVIDENCE'
            elif np.abs(p_value*2) < 0.05:
                evidence = 'STRONG EVIDENCE'
                num_strong += 1
            elif np.abs(p_value*2) < 0.1:
                evidence = 'WEAK EVIDENCE'
                num_weak += 1
            else:
                evidence = 'NO EVIDENCE'
                num_no += 1

            if t_test > 0:
                adjective = features_top[feat_idx]
            else:
                adjective = features_bottom[feat_idx]

            print(f'Mandolin {labels[idx]} - {evidence} {adjective.upper()}')

    # print(f'num strong {num_strong}')
    # print(f'num weak {num_weak}')
    # print(f'num no {num_no}')

    print()

    import matplotlib.pyplot as plt
    plt.hist(np.array(t_all))
    plt.show()
    plt.hist(np.array(p_all), 500)
    plt.xlim(0, 0.02)
    plt.show()

if __name__ == '__main__':

    # SURVEY PART 2
    csv_path = '../data/part2_111.csv'
    data = prepare_data(csv_path, std_min=1)

    main(data)

    print()

