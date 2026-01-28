from math import log2
def extract_features(psg, annotations, lights_off):
    features = {}
    features['Total_Sleep_Time'] = (annotations.count('N1') + annotations.count('N2') + annotations.count('N3') + annotations.count('N4') + annotations.count('R'))*30 - lights_off
    n = len(annotations)
    l = n
    for i in annotations[::-1]:
        if i != 'W':
            break
        else:
            l -= 1
    features['Total_Time_In_Bed'] = (l-1)*30
    features['Sleep_Efficiency'] = (features['Total_Sleep_Time'] / features['Total_Time_In_Bed'])*100 if features['Total_Time_In_Bed'] > 0 else 0
    sleep_onset = min(annotations.index('N1'), annotations.index('N2'), annotations.index('N3'), annotations.index('N4'), annotations.index('R'))
    features['Sleep_Latency'] = sleep_onset*30 - lights_off
    features['Rem_Latency'] = annotations.index('R')*30 - sleep_onset*30
    features['Wake_After_Sleep_Onset'] = annotations[sleep_onset:l].count('W')*30
    features['N1%'] = annotations.count('N1') / features['Total_Sleep_Time']*100
    features['N2%'] = annotations.count('N2') / features['Total_Sleep_Time']*100
    features['N3%'] = annotations.count('N3') / features['Total_Sleep_Time']*100
    features['N4%'] = annotations.count('N4') / features['Total_Sleep_Time']*100
    features['REM%'] = annotations.count('R') / features['Total_Sleep_Time']*100
    features['REM_NREM%'] = (features['REM%']/(features['N1%'] + features['N2%'] + features['N3%'] + features['N4%'])) *100
    features['Sleep_Stage_Transitions_Count'] = 0
    P = {
        'W': {"W": 0, 'N1': 0, 'N2': 0, 'N3': 0, 'N4': 0, 'R': 0},
        'N1': {"W": 0, 'N1': 0, 'N2': 0, 'N3': 0, 'N4': 0, 'R': 0},
        'N2': {"W": 0, 'N1': 0, 'N2': 0, 'N3': 0, 'N4': 0, 'R': 0},
        'N3': {"W": 0, 'N1': 0, 'N2': 0, 'N3': 0, 'N4': 0, 'R': 0},
        'N4': {"W": 0, 'N1': 0, 'N2': 0, 'N3': 0, 'N4': 0, 'R': 0},
        'R': {"W": 0, 'N1': 0, 'N2': 0, 'N3': 0, 'N4': 0, 'R': 0}
    }
    for i in range(n-1):
        if annotations[i] != annotations[i+1]:
            features['Sleep_Stage_Transitions_Count'] += 1
            P[annotations[i]][annotations[i+1]] += 1
    entropy = 0
    for i in P:
        for j in P[i]:
            P[i][j] = P[i][j] / features['Sleep_Stage_Transitions_Count']
            if P[i][j] > 0:
                entropy += P[i][j] * log2(P[i][j])
    features['Stage_Transition_Entropy'] = -entropy
    