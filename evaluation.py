import csv
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
from loraEnv import loraEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, TD3, SAC

def combine_csv_directory(local_dir):
    files = os.path.join(local_dir, "*.csv")
    files = glob.glob(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv(local_dir + "total.csv")
    return df

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Load model
model = PPO.load("logs/best_model.zip") # para cargar modelo que se haya generado en el solver
# TODO: Test other algorithms too

BER = [0.00013895754823009532, 6.390550739301948e-05, 2.4369646975025416e-05, 7.522516546093483e-06,
       1.8241669079988032e-06, 3.351781950877708e-07]

BER_NORM = normalize_data(BER)

env = loraEnv(1)
state = env.reset()

# TODO: Incluyo esta funcion para representar, ahorra mucho codigo despues (ver lo comentado....)
def step_csv(state,ber_th):
    header = ['ber', 'ber_th', 'battery_life', 'prr', 'SF']
    with open('results/'+str(i)+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        state = env.set_ber(ber_th)
        for k in range(100):
            action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
            env.step(action)
            ber, ber_norm, duration, prr, state = env.getStatistics()
            ber_th = BER[int(np.where(BER_NORM == ber_norm)[0])]
            data_row = [ber, ber_th, duration, prr, state]
            writer.writerow(data_row)

# EVALUATE AND SAVE RESULTS
for i, ber_th in enumerate(BER_NORM):
    step_csv(state, ber_th)

# PLOT
def plot_ber_comparison(combine):
    local_dir = 'results/'
    if combine:
        combine_csv_directory(local_dir)
    data = pd.read_csv(local_dir + 'total.csv')
    fig, ax1 = plt.subplots(1, figsize=(8, 5))
    sns.set_style("dark")
    sns.barplot(x='ber_th', y='ber', data=data,
                palette='deep', capsize=0.05, errwidth=1, ax=ax1).set(xlabel='Target BER', ylabel='BER')
    for i, ber_th in enumerate(BER[::-1]):
        plt.hlines(y=ber_th, xmin=i-0.5, xmax=0.5+i, color='red', alpha=0.3)
    ax1.set_xticklabels(["3.35e-07", "1.82e-06", "7.52e-06", "2.43e-05", "6.39e-05", "0.000138"], fontsize=8)
    plt.tight_layout()
    plt.savefig('results/comparison_ber_th.png', dpi=400)
    plt.show()

combine = True
plot_ber_comparison(combine)