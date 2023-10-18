import csv
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
from new import loraEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
import decimal
import random
from scipy.stats import loguniform
import math
from scipy import special as sp
from sb3_contrib import RecurrentPPO

VOLTAGE = 3  # V
CUT_OFF_VOLTAGE = 2.2  # V

def qfunc(x):
    return 0.5 - 0.5 * sp.erf(x / math.sqrt(2))

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


BER_TH = [1.6219899624731182e-14, 7.723365443074971e-11, 1.1145322639338472e-06, 7.130405140623094e-05,
          4.118253144853812e-13, 6.151947614848303e-04, 8.33448400134252e-7, 8.736955178359599e-10]

SNR = -6

DISTANCE = 2

REAL_DISTANCE = [2, 4, 6, 8]

local_dir = 'results/scenario1/'


env = loraEnv(1)

model_a2c = A2C.load("logs/lora_rl_a2c_vuibk.zip")
model_a2c = A2C.load(os.path.join("logs/lora_rl_a2c_vuibk.zip"), env=env,
                     custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})


def evaluation(local_dir, algorithm, model):
    header = ['ber', 'ber_th', 'snr', 'snr_measured', 'distance', 'distance_th', 'duration', 'state', 'pt',
              'energy_cons', 'e', 'sf']
    with open(local_dir + algorithm + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        env.reset()
        for j, ber_th in enumerate(BER_TH):
            state = env.set_ber_snr_distance(BER_TH[j], SNR, DISTANCE, 1)
            for k in range(1):
                if algorithm == "SF7":
                    action = (5, 0)
                    env.step(action)
                elif algorithm == "SF12":
                    action = (0, 0)
                    env.step(action)
                else:
                    action, _state = model.predict(state)
                    env.step(action)
                ber, ber_th, snr, snr_measured, distance, distance_th, duration, state, pt, energy, N, ber_diff, \
                distance_diff, energy_cons, sf = env.getStatistics()
                data_row = [ber, ber_th, snr, snr_measured, distance, distance_th, duration, state, pt, energy_cons, energy, sf]
                writer.writerow(data_row)


def evaluation_distances(local_dir, algorithm, model):
    header = ['ber', 'ber_th', 'snr', 'snr_measured', 'distance', 'distance_th', 'duration', 'state', 'pt', 'energy',
              'N', 'ber_diff', 'distance_diff']
    for i, d in enumerate(REAL_DISTANCE):
        with open(local_dir + 'd_' + str(d) + '_' + algorithm + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            env.reset()
            for j, ber_th in enumerate(BER_TH):
                state = env.set_ber_snr_distance(BER_TH[j], SNR, REAL_DISTANCE[i], 2)
                for k in range(100):
                    if algorithm == "SF7":
                        action = (5, 0)
                        env.step(action)
                    elif algorithm == "SF12":
                        action = (0, 0)
                        env.step(action)
                    else:
                        action, _state = model.predict(state)
                        env.step(action)
                    ber, ber_th, snr, snr_measured, distance, distance_th, duration, state, pt, energy, N, ber_diff, \
                        distance_diff = env.getStatistics()
                    data_row = [ber, ber_th, snr, snr_measured, distance, distance_th, duration, state, pt, energy, N,
                                ber_diff, distance_diff]
                    writer.writerow(data_row)

def evaluation_2():
    evaluation(local_dir, "A2C", model_a2c)
    #evaluation(local_dir, "SF7", model_ppo)
    #evaluation(local_dir, "SF12", model_ppo)

evaluation_2()


d=2
#data_ppo = pd.read_csv(local_dir + 'd_' + str(d) + '_PPO.csv')
#data_SF7 = pd.read_csv(local_dir + 'd_' + str(d) + '_SF7.csv')
#data_SF12 = pd.read_csv(local_dir + 'd_' + str(d) + '_SF12.csv')
data_ppo = pd.read_csv(local_dir + 'A2C.csv')
data_SF7 = pd.read_csv(local_dir + 'SF7.csv')
data_SF12 = pd.read_csv(local_dir + 'SF12.csv')


data = [data_ppo, data_SF7, data_SF12]
colors = ['tab:blue', 'tab:brown', 'tab:orange']
labels = ["DPLN", "Max", "Min"]
distances = [2, 4, 6, 8]


def plot_ber_comparison():
    fig0 = plt.figure(figsize=(5, 4))
    ax0 = fig0.subplots(1, 1)

    for i, d in enumerate(data):
        ber = smooth(d['ber'], 10)
        ber = np.clip(ber, 1e-16, None)
        if i >= 1:
            sns.lineplot(ax=ax0, x=d.index, y=ber, data=d, label=labels[i], alpha=.6, color=colors[i], ls=':')
        else:
            sns.lineplot(ax=ax0, x=d.index, y=ber, data=d, label=labels[i], alpha=.7, color=colors[i])
    sns.lineplot(ax=ax0, x=data_ppo.index, y=data_ppo['ber_th'], data=data_ppo, label='$BER_{max}$', alpha=1, color='red')
    ax0.legend()
    ax0.set(xlabel='Uplink messages', ylabel='BER')
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax0.set_yscale('log')
    ax0.yaxis.grid(True, linestyle='-', alpha=0.4)
    fig0.tight_layout()
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('results/s1_ber.pdf', dpi=400, format='pdf')
    plt.show()


def plot_ber_energy():
    ranges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    magnitudes = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5]
    alpha = [0.6, 0.6, 0.6]
    shift = [-0.25, 0.25]
    fill_pattern = ['', '//', '//']

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    for i, d in enumerate(data):
        ber = d['ber']
        ber_th = d['ber_th']
        energy = d['energy_cons']

        size_range = 11

        for j, r in enumerate(ranges):
            inicio = j * size_range
            fin = (j + 1) * size_range
            ber = np.clip(ber, 1e-16, None)
            ber_mean = ber[inicio:fin].mean()
            ber_th_mean = ber_th[inicio:fin].mean()
            energy_mean = energy[inicio:fin].mean()

            if i >= 1:
                pass
            else:
                ax0.bar(r + shift[i], ber_mean, width=0.50, zorder=5, color=colors[i],
                        edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                        label=labels[i] if j == 0 else "", fill=True)

                ax0.bar(r + shift[i+1], ber_th_mean, width=0.50, zorder=5, color='none',
                        edgecolor='red', alpha=1, lw=1., label='$BER_{max}$' if j == 0 else "", fill=True)

                ax1.bar(r + 0, energy_mean, width=0.50, zorder=5, color=colors[i],
                        edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                        label=labels[i] if j == 0 else "", fill=True)

    ax0.legend(fontsize=9)
    ax0.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax0.set(ylabel='BER')
    ax0.set_yscale('log')
    ax0.set_xticks([])

    ax1.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax1.set(xlabel='Order of magnitude of the desired $BER_{max}$', ylabel='Energy consumption (J)')
    ax1.set_xticks(ranges)
    ax1.set_xticklabels(magnitudes)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    plt.savefig('results/s1_ber_energy.pdf', dpi=400, format='pdf')
    plt.show()


def plot_distance():
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.subplots(1, 1)

    distances = [2, 4, 6, 8]
    alpha = [0.6, 0.6, 0.6]
    shift = [-0.35, 0, 0.35]
    fill_pattern = ['', '//', '//']

    for i, dist in enumerate(distances):
        data_ppo = pd.read_csv(local_dir + 'd_' + str(dist) + '_PPO.csv')
        data_SF7 = pd.read_csv(local_dir + 'd_' + str(dist) + '_SF7.csv')
        data_SF12 = pd.read_csv(local_dir + 'd_' + str(dist) + '_SF12.csv')

        data = [data_ppo, data_SF7, data_SF12]
        for j, d in enumerate(data):
            mean_d_diff = (np.mean(d["distance_diff"]))
            std_error_d = np.std(d["distance_diff"], ddof=1) / np.sqrt(len(d["distance_diff"]))
            if j >= 1:
                ax1.bar(dist + shift[j], mean_d_diff, width=0.35, zorder=5, color='none',
                        edgecolor=colors[j], alpha=alpha[j], hatch=fill_pattern[j], lw=1., label=labels[j] if i == 0 else "")

            else:
                ax1.bar(dist + shift[j], mean_d_diff, yerr=std_error_d,
                        error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), capsize=2, width=0.35, zorder=5,
                        color=colors[j], alpha=alpha[j], hatch=fill_pattern[j], label=labels[j] if i == 0 else "")
    ax1.set_xticks(distances)
    ax1.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax1.set_ylabel(r'$d_{max} - d$ (km)')
    ax1.set_xlabel('Measured distance $d$ (km)')
    ax1.legend(ncol=2, loc='upper center', fontsize=9)
    #ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #ax1.set_yscale('log')
    plt.tight_layout()
    plt.savefig('results/s1_distance.pdf', dpi=400, format='pdf')
    plt.show()



plot_ber_comparison()
plot_ber_energy()
plot_distance()

