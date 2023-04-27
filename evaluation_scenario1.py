import csv
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
from loraEnv import loraEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
import math


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# INIT
max_target_BER = 0.00013926357210214731
min_target_BER = 5.378212320056908e-07
SNR = 0.06309573445
BER_TH = [0.00013895754823009532, 6.390550739301948e-05, 2.4369646975025416e-05, 7.522516546093483e-06,
          1.8241669079988032e-06, 3.351781950877708e-07]

REAL_DISTANCE = [2, 4, 6, 8]
env = loraEnv(10)
state = env.reset()
local_dir = 'results/scenario1/'


def increasing_evaluation(local_dir, algorithm, model):
    header = ['ber', 'ber_th', 'snr', 'snr_db', 'snr_measured', 'snr_measured_db', 'distance', 'distance_th', 'duration',
              'energy_cons', 'prr', 'pdr', 'N', 'SF', 'pt', 'ber_diff', 'distance_diff']
    for i, d in enumerate(REAL_DISTANCE):
        with open(local_dir + 'd_' + str(d) + '_' + algorithm + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            env.reset()
            for j, ber_th in enumerate(BER_TH):
                state = env.set_ber_distance_snr(BER_TH[j], SNR, REAL_DISTANCE[i], 1)
                for k in range(100):
                    if algorithm == "SF7":
                        action = 0
                        env.step(action)
                    elif algorithm == "SF12":
                        action = 2
                        env.step(action)
                    else:
                        action, _state = model.predict(state)
                        env.step(action)
                    ber, ber_th, snr, snr_th, distance, distance_th, duration, c_total, prr, pdr, N, SF, e, pt = env.getStatistics()
                    snr_db = 10 * math.log10(snr)
                    snr_th_db = 10 * math.log10(snr_th)
                    #ber_th = BER_TH[int(np.where(BER_TH_NORM == ber_norm)[0])]
                    #distance_th = REAL_DISTANCE[int(np.where(REAL_DISTANCE_NORM == distance_norm)[0])]
                    data_row = [ber, ber_th, snr, snr_db, snr_th, snr_th_db, distance, distance_th, duration, c_total,
                                prr, pdr, N, SF, pt, ber - ber_th, distance - distance_th]
                    writer.writerow(data_row)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# Load model
model_ppo = PPO.load("logs/best_model.zip")
#model_a2c = A2C.load("logs/best_model_a2c.zip")
#model_rec = RecurrentPPO.load("logs/best_model_rec.zip")


def evaluate():
    increasing_evaluation(local_dir, "PPO", model_ppo)
    #increasing_evaluation(local_dir, "A2C", model_a2c)
    #increasing_evaluation(local_dir, "RecurrentPPO", model_rec)
    #increasing_evaluation(local_dir, "SF7", model_rec)
    #increasing_evaluation(local_dir, "SF12", model_rec)


# > EVALUATE - Uncomment to evaluate and get results again
# > Results will be stored in results/scenario1 folder
evaluate()

d = 2
data_ppo = pd.read_csv(local_dir + 'd_' + str(d) + '_PPO.csv')
#data_a2c = pd.read_csv(local_dir + 'd_' + str(d) + '_A2C.csv')
#data_rec = pd.read_csv(local_dir + 'd_' + str(d) + '_RecurrentPPO.csv')
#data_SF7 = pd.read_csv(local_dir + 'd_' + str(d) + '_SF7.csv')
#data_SF12 = pd.read_csv(local_dir + 'd_' + str(d) + '_SF12.csv')

data = [data_ppo]
print(data_ppo['ber'])
#data = [data_ppo, data_a2c, data_rec, data_SF7, data_SF12]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink']
labels = ["PPO", "A2C", "RecPPO", "Min", "Max"]


def plot_energy_comparison():
    fig1 = plt.figure(figsize=(5, 4))
    ax1 = fig1.subplots(1, 1)
    for i, d in enumerate(data):
        battery = d['energy'] * 1000
        if i >= 3:
            sns.lineplot(ax=ax1, x=d.index, y=battery, data=d, label=labels[i], alpha=.5, color=colors[i], ls=':')
        else:
            sns.lineplot(ax=ax1, x=d.index, y=battery, data=d, label=labels[i], alpha=.5, color=colors[i])
    ax1.legend()
    ax1.set(xlabel='Uplink messages', ylabel='Battery (mAh)')
    fig1.tight_layout()
    plt.savefig('results/s1_energy.png', dpi=400)
    plt.show()


def plot_ber_comparison():
    fig0 = plt.figure(figsize=(5, 4))
    ax0 = fig0.subplots(1, 1)
    for i, d in enumerate(data):
        ber = smooth(d['ber'], 10)
        if i >= 3:
            sns.lineplot(ax=ax0, x=d.index, y=ber, data=d, label=labels[i], alpha=.5, color=colors[i], ls=':')
        else:
            sns.lineplot(ax=ax0, x=d.index, y=ber, data=d, label=labels[i], alpha=.5, color=colors[i])
    sns.lineplot(ax=ax0, x=d.index, y='ber_th', data=d, label=r'$BER_{th}$', alpha=1, color='red')
    ax0.legend()
    ax0.set(xlabel='Uplink messages', ylabel='BER')
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax0.set_yscale('log')
    fig0.tight_layout()
    plt.savefig('results/s1_ber.png', dpi=400)
    plt.show()


def plot_ber_bars_distances():
    fig = plt.figure(figsize=(5, 4))
    ax0 = fig.subplots(1, 1)

    distances = [2, 4, 6, 8]
    alpha = [0.6, 0.6, 0.6, 0.6, 0.6]
    shift = [-0.5, -0.25, 0, 0.25, 0.5]

    for i, dist in enumerate(distances):
        data_ppo = pd.read_csv(local_dir + 'd_' + str(dist) + '_PPO.csv')
        data_a2c = pd.read_csv(local_dir + 'd_' + str(dist) + '_A2C.csv')
        data_rec = pd.read_csv(local_dir + 'd_' + str(dist) + '_RecurrentPPO.csv')
        data_SF7 = pd.read_csv(local_dir + 'd_' + str(dist) + '_SF7.csv')
        data_SF12 = pd.read_csv(local_dir + 'd_' + str(dist) + '_SF12.csv')

        data = [data_ppo, data_a2c, data_rec, data_SF7, data_SF12]
        for j, d in enumerate(data):
            mean_ber_diff = np.abs(np.mean(d["ber_diff"]))
            std_error_ber = np.std(d["ber_diff"], ddof=1) / np.sqrt(len(d["ber_diff"]))

            ax0.bar(dist + shift[j], mean_ber_diff, yerr=std_error_ber,
                    error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), capsize=2, width=0.25, zorder=5,
                    color=colors[j], alpha=alpha[j])
            ax0.plot([], [], lw=5, color=colors[j])
    ax0.set_xticks(distances)
    ax0.set_ylabel(r'$BER - BER_{th}$')
    ax0.set_xlabel('Real distance node-gateway (km)')
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()

    plt.savefig('results/s1_ber_diff_distances.png', dpi=400)
    plt.show()


def plot_sth_else():
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.subplots(1, 1)

    distances = [2, 4, 6, 8]
    alpha = [0.6, 0.6, 0.6, 0.6, 0.6]
    shift = [-0.5, -0.25, 0, 0.25, 0.5]

    for i, dist in enumerate(distances):
        data_ppo = pd.read_csv(local_dir + 'd_' + str(dist) + '_PPO.csv')
        data_a2c = pd.read_csv(local_dir + 'd_' + str(dist) + '_A2C.csv')
        data_rec = pd.read_csv(local_dir + 'd_' + str(dist) + '_RecurrentPPO.csv')
        data_SF7 = pd.read_csv(local_dir + 'd_' + str(dist) + '_SF7.csv')
        data_SF12 = pd.read_csv(local_dir + 'd_' + str(dist) + '_SF12.csv')

        data = [data_ppo, data_a2c, data_rec, data_SF7, data_SF12]
        for j, d in enumerate(data):
            mean_d_diff = np.abs(np.mean(d["distance_diff"]))
            std_error_d = np.std(d["distance_diff"], ddof=1) / np.sqrt(len(d["distance_diff"]))
            ax1.bar(dist + shift[j], mean_d_diff, yerr=std_error_d,
                    error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), capsize=2, width=0.25, zorder=5,
                    color=colors[j], alpha=alpha[j])
            ax1.plot([], [], lw=5, color=colors[j])
    ax1.set_xticks(distances)
    ax1.set_ylabel(r'$d_{GW} - d_{MAX}$')
    ax1.set_xlabel('Real distance node-gateway (km)')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(local_dir + 's1_sth_else.png', dpi=400)
    plt.show()


# > PLOT - Uncomment to plot figures
plot_ber_comparison()
# plot_energy_comparison()
#plot_ber_bars_distances()
# plot_sth_else()