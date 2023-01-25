# coding=utf-8
import csv
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
from loraEnv import loraEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, TD3, SAC

nodes = 5

def combine_csv_directory(local_dir):
    """
    Not used anymore
    :param local_dir:
    :return:
    """
    files = os.path.join(local_dir, "*.csv")
    files = glob.glob(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv(local_dir + "total.csv")
    return df


def increasing_evaluation(local_dir):
    """
    This function simply evaluates the training environment using increasing BET_th and real distance measured over time
    :param local_dir:
    :return:
    """
    # EVALUATE AND SAVE RESULTS
    header = ['ber', 'ber_th', 'distance', 'distance_th', 'battery_life', 'prr', 'pdr', 'N', 'state', 'algorithm']
    with open(local_dir+'main_results.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, ber_th in enumerate(BER_TH_NORM):
            state = env.set_ber_distance(ber_th, REAL_DISTANCE_NORM[i], nodes)
            for k in range(100):
                action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
                env.step(action)
                ber, ber_norm, distance, distance_norm, duration, prr, pdr, N, state = env.getStatistics()
                ber_th = BER_TH[int(np.where(BER_TH_NORM == ber_norm)[0])]
                distance_th = REAL_DISTANCE[int(np.where(REAL_DISTANCE_NORM == distance_norm)[0])]
                algorithm = 'PPO'
                data_row = [ber, ber_th, distance, distance_th, duration, prr, pdr, N, state, algorithm]
                writer.writerow(data_row)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Load model
model = A2C.load("logs/best_model.zip")  # para cargar modelo que se haya generado en el solver

BER_TH = [0.00013895754823009532, 6.390550739301948e-05, 2.4369646975025416e-05, 7.522516546093483e-06,
          1.8241669079988032e-06, 3.351781950877708e-07]

BER_TH_NORM = normalize_data(BER_TH)

REAL_DISTANCE = [2.6179598590188147, 3.2739303314239954, 4.094264386099205, 5.12014586944165, 6.403077879720777,
                 8.00746841578568]

REAL_DISTANCE_NORM = normalize_data(REAL_DISTANCE)

env = loraEnv(10)
state = env.reset()

# EVALUATE
##############################

local_dir = 'results/'
increasing_evaluation(local_dir)
data = pd.read_csv(local_dir + 'main_results.csv')


# PLOT BER
##############################
def plot_ber_bars():
    fig, ax1 = plt.subplots(1, figsize=(8, 5))
    sns.barplot(x='ber_th', y='ber', data=data,
                palette='deep', capsize=0.05, errwidth=1, ax=ax1).set(xlabel='Target BER', ylabel='BER')
    for i, ber_th in enumerate(BER_TH[::-1]):
        plt.hlines(y=ber_th, xmin=i-0.5, xmax=0.5+i, color='red', alpha=0.3)
    ax1.set_xticklabels(["3.35e-07", "1.82e-06", "7.52e-06", "2.43e-05", "6.39e-05", "0.000138"], fontsize=8)
    ax1.set_title(str(nodes) + ' nodes')
    plt.tight_layout()
    plt.savefig(local_dir+'ber_bars.png', dpi=400)
    plt.show()


def plot_ber_lines():
    # EVALUATE AND SAVE RESULTS
    fig, ax1 = plt.subplots(1, figsize=(8, 5))
    sns.lineplot(x=data.index, y='ber', data=data, label="Measured BER", alpha=.5, color='blue', ax=ax1)
    sns.lineplot(x=data.index, y='ber_th', data=data, label="Target BER", alpha=.5,
                 color='red', ax=ax1).set(xlabel='Uplink messages', ylabel='BER')
    plt.tight_layout()
    plt.legend()
    plt.savefig(local_dir+'ber_lines.png', dpi=400)
    plt.title(str(nodes) + ' nodes')
    plt.show()


# PLOT DISTANCE
##############################
def plot_distance_bars():
    fig, ax1 = plt.subplots(1, figsize=(8, 5))
    sns.barplot(x='distance_th', y='distance', data=data,
                palette='deep', capsize=0.05, errwidth=1, ax=ax1).set(xlabel='Target BER', ylabel='BER')
    for i, ber_th in enumerate(BER_TH[::-1]):
        plt.hlines(y=ber_th, xmin=i-0.5, xmax=0.5+i, color='red', alpha=0.3)
    ax1.set_xticklabels(["3.35e-07", "1.82e-06", "7.52e-06", "2.43e-05", "6.39e-05", "0.000138"], fontsize=8)
    ax1.set_title(str(nodes) + ' nodes')
    plt.tight_layout()
    plt.savefig(local_dir+'distance_bars.png', dpi=400)
    plt.show()


def plot_distance_lines():
    # EVALUATE AND SAVE RESULTS
    fig, ax1 = plt.subplots(1, figsize=(8, 6))
    sns.lineplot(x=data.index, y='distance', data=data, label="Estimated Maximum distance", alpha=.5, color='blue', ax=ax1)
    sns.lineplot(x=data.index, y='distance_th', data=data, label="Real distance", alpha=.5,
                 color='red', ax=ax1).set(xlabel='Uplink messages', ylabel='Distance (Km)')
    plt.tight_layout()
    plt.legend()
    plt.savefig(local_dir+'distance_lines.png', dpi=400)
    plt.title(str(nodes) + ' nodes')
    plt.show()


plot_ber_lines()
plot_distance_lines()
