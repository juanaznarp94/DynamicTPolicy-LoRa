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


BER_TH = [1.4217740977130744e-14, 5.2754709005326995e-14, 1.6066086796644992e-14,3.238764226811042e-14, 7.791583179062107e-14,7.155374960182833e-14, 1.583215738264315e-14, 6.015349718800581e-14, 2.1212749109013486e-14, 5.206851577661693e-14, 6.033230799777314e-14,#
          8.736955178359599e-13, 2.1969026450552186e-13, 1.0355095220641802e-13, 8.479367880103934e-13, 2.536564730776935e-13, 1.7659337040379086e-13, 3.8127454970826717e-13, 1.5222119996894606e-13, 6.921613591714081e-13, 5.688799052363484e-13, 2.3060056417216496e-13,
          7.865013798485343e-12, 8.454535561744325e-12, 3.4907210824362274e-12, 4.4392176941533505e-12, 1.9642146413299937e-12, 1.2211987633943662e-12, 6.733485745098573e-12, 7.548055278149162e-12, 1.031594447915946e-12, 6.3764018765809575e-12, 3.954311157913585e-12,
          5.0000152478990106e-11, 1.1804868635610476e-11, 1.471865230543367e-11, 4.04300879159477e-11, 7.985845349507768e-11, 4.933293112154284e-11, 1.1128084536227304e-11, 2.4545659937314522e-11, 3.7291191859027e-11, 2.8342203928583386e-11, 4.173575820441247e-11,
          1.5650141273896386e-10, 2.5677305039442233e-10, 9.442946199118856e-10, 3.862705400175048e-10, 6.094310803863996e-10, 1.9616255766979685e-10, 7.203467221516637e-10, 2.2473886854150493e-10, 5.374348233739061e-10, 6.891941538038956e-10, 1.786620312228757e-10,
          1.5464033575899254e-09, 3.801115121561928e-09, 2.6869071239332605e-09, 1.986269791016766e-09, 9.548693830315354e-09, 3.3777089052380033e-09, 4.71978034116666e-09, 6.0260014182524765e-09, 2.739060231834956e-09, 1.8239712905915384e-09, 3.902675668062559e-09,
          1.5698512358391936e-08, 7.472571338676683e-08, 5.557964289754211e-08, 1.506972334241027e-08,2.2375693007303908e-08, 3.712436473109554e-08, 1.1759655029505868e-08, 3.007657211655298e-08, 1.1443910124988297e-08, 4.669931146851406e-08, 6.01653364784323e-08,
          6.1515338583209209e-07, 5.8563739858681e-07, 7.1720423918271317e-07, 3.078176161740871e-07, 1.1263874085226397e-07, 9.31418464328964e-07, 3.887172328455135e-07, 2.4258918258180406e-07, 7.194842746874597e-07, 5.891978477080726e-07, 8.94769964813225e-07,
          1.1415449205238442e-06, 2.3336730217510854e-06, 2.7545154604463345e-06, 1.1387382179874704e-06, 1.5405261655278632e-06, 3.901192032891095e-06, 2.0059945456119368e-06, 5.32091277751163e-06, 2.041305922719148e-06, 8.506722412948916e-06, 5.9081795976300045e-06,
          5.278963329866298e-05, 3.516716324669859e-05, 7.803715143799452e-05, 5.286898197828745e-05, 2.187641331976098e-05, 1.743978924748565e-05, 1.3343880436180204e-05, 9.296288487394476e-05, 1.3765220202709633e-05, 6.236069916807322e-05, 7.10530053695712e-05]

#BER_TH = [1.6219899624731182e-14, 7.723365443074971e-11, 1.1145322639338472e-06, 7.130405140623094e-05,
#          4.118253144853812e-13, 6.151947614848303e-04, 8.33448400134252e-7, 8.736955178359599e-10]

SNR = -6

DISTANCE = 2

REAL_DISTANCE = [2, 4, 6, 8]

local_dir = 'results/scenario1/'


env = loraEnv(1)

model_a2c = A2C.load("logs/lora_rl_a2c_vuibk.zip")
#model_a2c = A2C.load("logs/best_model_a2c.zip")
#model_rec = RecurrentPPO.load("logs/best_model_rec.zip")


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
data_SF7 = pd.read_csv(local_dir + 'SF7nuevo2.csv')
data_SF12 = pd.read_csv(local_dir + 'SF12nuevo2.csv')


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
#plot_ber_energy()
#plot_distance()

