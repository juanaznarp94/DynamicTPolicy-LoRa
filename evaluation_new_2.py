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
from numpy import sin, cos, arccos, pi, round
from scipy.stats import rayleigh
from sb3_contrib import RecurrentPPO

# INIT
latitud_gw = 47.66577
longitud_gw = -122.34737
num_nodes = [1, 3, 5]
ALLOWED_TPS = [0.012589, 0.025119, 0.1]
MAX_BATTERY_LEVEL = 21312  # J
c = 3 * (10**8)  # speed of light (m/s)
frec = 868 * (10**6)  # LoRa frequency Europe (Hz)
pt = 14  # Transmission power (W)
sf = [7, 8, 9, 10, 11, 12]
snr_0 = 32  # value in lineal for SX1272 transceiver
nf = 4  # Noise figure in lineal (6 dB)
k = 1.380649 * (10**-23)  # Boltzmann constant (J/K)
t = 278  # temperature (K)
n_ = 3.1  # path loss exponent in urban area (2.7-3.5)
bw = [125000, 125000, 125000, 125000, 125000, 125000]  # Hz
ALLOWED_TPS = [0.012589, 0.025119, 0.1]
magnitudes = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5]

def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees


def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians


def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2, unit='kilometers'):
    theta = longitude1 - longitude2

    distance = 60 * 1.1515 * rad2deg(
        arccos(
            (sin(deg2rad(latitude1)) * sin(deg2rad(latitude2))) +
            (cos(deg2rad(latitude1)) * cos(deg2rad(latitude2)) * cos(deg2rad(theta)))
        )
    )

    if unit == 'miles':
        return round(distance, 2)
    if unit == 'kilometers':
        return round(distance * 1.609344, 2)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def qfunc(x):
    return 0.5 - 0.5 * sp.erf(x / math.sqrt(2))


df = pd.read_csv("trajectory_gps.csv", sep=',', header=0)  # load dataset
data = df[15:102]  # choosing a trajectory of 8 km
distances = []

for i in range(len(data)):
    longitud = data.iloc[i]['Longitude']
    latitud = data.iloc[i]['Latitude']
    dist = getDistanceBetweenPointsNew(latitud_gw, longitud_gw, latitud, longitud)
    distances.append(dist)

distances[0] = 0.05
SNR_TH = []

BER_TH = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05]

distances = []

for _ in range(88):
    random_number = round(random.uniform(6, 8), 2)
    distances.append(random_number)


#SNR_TH = [11, 5, 0, -3, -7, -11, -14, -18]
#SNR_TH = [5, 0, -4, -8, -13, -15, -17, -19]

local_dir = 'results/scenario2/'

env = loraEnv(1)

model_ppo = PPO.load("logs/best_model.zip")


def evaluation(local_dir, algorithm, model, ber_th):
    header = ['ber', 'ber_th', 'snr', 'snr_measured', 'distance', 'distance_th', 'duration', 'state', 'pt',
              'energy_cons', 'e', 'sf']
    snr_db_measured = 40
    #for n, nodes in enumerate(num_nodes):
    with open(local_dir + str(ber_th) + '_d6_.csv', 'w', encoding='UTF8', newline='') as f:
    #with open(local_dir + 'PPO_prueba.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        env.reset()
        action = (0, 0)
        env.step(action)
        for i, dist in enumerate(distances):
            pt = action[1]
            pt_w = ALLOWED_TPS[pt]
            Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
            snr_meas = pt_w / (Lpath * nf * k * t * bw[0])
            snr_db_measured = 10 * math.log10(snr_meas) + 1
            if algorithm == "SF7":
                action = (5, 0)
                pt = action[1]
                pt_w = ALLOWED_TPS[pt]
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = pt_w / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                env.step(action)
            elif algorithm == "SF12":
                action = (0, 0)
                pt = action[1]
                pt_w = ALLOWED_TPS[pt]
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = pt_w / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                env.step(action)
            else:
                state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                action, _state = model_ppo.predict(state)  # predecimos la acción más recomendada para ese estado
                env.step(action)

            ber, ber_max, snr, snr_db, distance, distance_th, duration, state, pt_d, energy, N, ber_diff, \
            distance_diff, energy_cons, sf = env.getStatistics()
            data_row = [ber, ber_th, snr, snr_db_measured, distance, distance_th, duration, state, ALLOWED_TPS[pt],
                        energy_cons, energy, sf]
            writer.writerow(data_row)


def evaluate(ber):
    evaluation(local_dir, "PPO", model_ppo, ber)
    #evaluation(local_dir, "A2C", model_a2c)
    #evaluation(local_dir, "RecurrentPPO", model_rec)
    #evaluation(local_dir, "SF7", model_ppo)
    #evaluation(local_dir, "SF12", model_ppo)


for i, ber in enumerate(BER_TH):
    evaluate(ber)

data_ppo = pd.read_csv(local_dir + 'n_1_PPO_prueba.csv')
data_ADR = pd.read_csv(local_dir + 'ADR.csv')
#data_SF7 = pd.read_csv(local_dir + 'n_1_SF7.csv')
#data_SF12 = pd.read_csv(local_dir + 'n_1_SF12.csv')


data = [data_ppo]
colors = ['tab:blue', 'tab:brown', 'tab:orange']
labels = ["DPLN", "Max", "Min"]


def equations(SNR_max, SNR_req, i, Pt, index_Pt):
    snr_margin = SNR_max - SNR_req - 10  # 1
    n_step = math.floor(snr_margin / 3)  # 2

    if n_step >= 0 and i != 5:
        i = i + 1
    elif n_step >= 0 and i == 5:
        i = i - 1
    elif n_step < 0 and index_Pt != 2:
        Pt = ALLOWED_TPS[index_Pt + 1]
    elif n_step < 0 and index_Pt == 2:
        pass
    return i, Pt


def activation_ADR(ber_th):
    header = ['ber', 'ber_th', 'snr', 'snr_measured', 'config', 'pt', 'energy_cons', 'sf']
    with open(local_dir + 'ADR_' + str(ber_th) + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        snr_db_measured = 46
        env.set_ber_nodes(ber_th, 1)
        i = 4
        Pt = 0.012589
        index_Pt = ALLOWED_TPS.index(Pt)
        action = (i, index_Pt)
        for j, dist in enumerate(distances):
            if j < 20:
                #INIT ADR
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)

            #ACTIVATE ADR 1
            if j == 20:
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                SNR_20_frames = SNR_TH[0:20]
                SNR_max = np.max(SNR_20_frames)
                SNR_req = SNR_20_frames[19]
                i, Pt = equations(SNR_max, SNR_req, i, Pt, index_Pt)
                index_Pt = ALLOWED_TPS.index(Pt)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            if j > 20 and j < 40:
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            #ACTIVATE ADR 2
            if j == 40:
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                SNR_20_frames = SNR_TH[20:40]
                SNR_max = np.max(SNR_20_frames)
                SNR_req = SNR_20_frames[19]
                i, Pt = equations(SNR_max, SNR_req, i, Pt, index_Pt)
                index_Pt = ALLOWED_TPS.index(Pt)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            if j > 40 and j < 60:
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            #ACTIVATE ADR 3
            if j == 60:
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                SNR_20_frames = SNR_TH[40:60]
                SNR_max = np.max(SNR_20_frames)
                SNR_req = SNR_20_frames[19]
                i, Pt = equations(SNR_max, SNR_req, i, Pt, index_Pt)
                index_Pt = ALLOWED_TPS.index(Pt)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            if j > 60 and j < 80:
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            #ACTIVATE ADR 4
            if j == 80:
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                SNR_20_frames = SNR_TH[60:80]
                SNR_max = np.max(SNR_20_frames)
                SNR_req = SNR_20_frames[19]
                i, Pt = equations(SNR_max, SNR_req, i, Pt, index_Pt)
                index_Pt = ALLOWED_TPS.index(Pt)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            if j > 80:
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            ber, ber_max, snr, snr_measured, distance, distance_th, duration, state, pt, energy, N, ber_diff, \
            distance_diff, energy_cons, sf = env.getStatistics()

            data_row = [ber, ber_th, snr, SNR_TH[j], state, pt, energy_cons, sf]
            writer.writerow(data_row)


#for i, ber in enumerate(BER_TH):
#    activation_ADR(ber)


def plot_snr_comparison():
    fig0 = plt.figure(figsize=(5, 4))
    ax0 = fig0.subplots(1, 1)

    for i, s in enumerate(data):
        if i >= 1:
            pass
        else:
            sns.lineplot(ax=ax0, x=s.index[0:80], y=data_ppo['snr'][0:80], data=s[0:80], label=labels[i], alpha=.7,
                         color=colors[i], ls='-')

    sns.lineplot(ax=ax0, x=data_ADR.index[0:80], y=data_ADR['snr'][0:80], data=data_ADR[0:80], label='ADR', alpha=0.7,
                 color='tab:red', ls='-')
    sns.lineplot(ax=ax0, x=data_ppo.index[0:80], y=data_ppo['snr_measured'][0:80], data=data_ppo[0:80],
                 label=r'$SNR_{measured}$', alpha=1, color='tab:blue', ls=':')
    sns.lineplot(ax=ax0, x=data_ADR.index[0:80], y=data_ADR['snr_measured'][0:80], data=data_ADR[0:80],
                 label=r'$SNR_{measured_{ADR}}$', alpha=1, color='tab:red', ls=':')

    # Agregar leyenda de colores
    legend_colors = ax0.legend(handles=[plt.Line2D([], [], color='tab:blue', linestyle='-'),
                                        plt.Line2D([], [], color='tab:red', linestyle='-')],
                               labels=['DPLN', 'ADR'], fontsize=9, loc=(0.775, 0.8))
    # Agregar leyenda de estilos
    legend_styles = ax0.legend(handles=[plt.Line2D([], [], color='black', linestyle='-'),
                                        plt.Line2D([], [], color='black', linestyle=':')],
                               labels=['SNR min', 'SNR'], fontsize=9, loc=(0.726, 0.63))

    # Añadir las leyendas a la gráfica
    ax0.add_artist(legend_colors)
    ax0.add_artist(legend_styles)
    ax0.set(xlabel='Uplink messages', ylabel='SNR (dB)')
    fig0.tight_layout()
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.savefig('results/s2_snr.pdf', dpi=400, format='pdf')
    plt.show()


def plot_distance_comparison():
    fig0 = plt.figure(figsize=(5, 4))
    ax0 = fig0.subplots(1, 1)

    for i, d in enumerate(data):
        #distance = smooth(d['distance'], 10)
        if i >= 1:
            pass
        else:
            sns.lineplot(ax=ax0, x=d.index[0:83], y=data_ppo['distance'][0:83], data=d[0:83],
                         label='Maximum distance DPLN $d_{max}$', alpha=.7, color=colors[i])

    sns.lineplot(ax=ax0, x=data_ppo.index[0:83], y=data_ppo['distance_th'][0:83], data=data_ppo[0:83],
                 label='Measured distance $d$', alpha=1, color='tab:blue', linestyle=':')

    legend_dpln = ax0.legend(handles=[plt.Line2D([], [], color='tab:blue', linestyle='-')],
                             labels=['Maximum distance DPLN $d_{max}$'], fontsize=9, loc=(0.45, 0.2))
    # Crear leyenda de la línea discontinua (Gateway-node distance)
    legend_distance = ax0.legend(handles=[plt.Line2D([], [], color='tab:blue', linestyle=':')],
                                 labels=['Measured distance $d$'], fontsize=9, loc=(0.58, 0.1))

    # Agregar leyendas a la gráfica
    ax0.add_artist(legend_dpln)
    ax0.add_artist(legend_distance)
    ax0.set(xlabel='Uplink messages', ylabel='Distance (km)')
    fig0.tight_layout()
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.savefig('results/s2_distance.pdf', dpi=400, format='pdf')
    plt.show()


def plot_cdf(data_ppo, data_ADR):

    # sort the data:
    data_sorted_ppo = np.sort(data_ppo)
    data_sorted_adr = np.sort(data_ADR)

    # calculate the proportional values of samples
    p_ppo = 1. * np.arange(len(data_ppo)) / (len(data_ppo) - 1)
    p_adr = 1. * np.arange(len(data_ADR)) / (len(data_ADR) - 1)

    # plot the sorted data:
    fig = plt.figure(figsize=(5, 4))
    plt.plot(data_sorted_ppo, p_ppo, color='tab:blue', label='DPLN')
    plt.plot(data_sorted_adr, p_adr, color='tab:green', label='ADR')

    plt.grid(True, linestyle='-', alpha=0.4)
    plt.legend(fontsize=9)
    plt.xlabel('Estimated BER')
    plt.ylabel('$p$')
    plt.savefig('s2_cdf.png', dpi=400)
    plt.show()


def plot_ber_energy_magnitudes():
    ranges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    magnitudes = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5]
    alpha = [0.6, 0.6, 0.6]
    shift = [-0.50, 0]
    fill_pattern = ['', '//', '//']
    labels = ['DPLN', 'ADR']
    colors = ['tab:blue', 'tab:red']

    fig = plt.figure(figsize=(6, 7))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    for j, mag in enumerate(magnitudes):
        data_ppo = pd.read_csv(local_dir + '1e' + str(mag) + '.csv')
        data_ADR = pd.read_csv(local_dir + 'ADR_1e' + str(mag) + '.csv')
        data = [data_ppo, data_ADR]
        for i, d in enumerate(data):
            ax0.bar(ranges[j] + shift[i], d['ber'].mean(), width=0.50, zorder=5, color=colors[i],
                    edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels[i] if j == 0 else "", fill=True)

            ax2.bar(ranges[j] + shift[i], d['energy_cons'].mean(), width=0.50, zorder=5, color=colors[i],
                    edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels[i] if j == 0 else "", fill=True)

            ax1.bar(ranges[j] + 0.5, d['snr'].mean(), width=0.50, zorder=5, color=colors[i],
                    edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels[i] if j == 0 else "", fill=True)

            ax1.bar(ranges[j] + shift[i], d['snr_measured'].mean(), width=0.50, zorder=5, color=colors[i],
                    edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels[i] if j == 0 else "", fill=True)

            if i == 0:
                ax0.bar(ranges[j] + 0.5, d['ber_th'].mean(), width=0.50, zorder=5, color='none',
                        edgecolor='red', alpha=1, lw=1., label='$BER_{max}$' if j == 0 else "", fill=True)
                ax1.bar(ranges[j] + shift[i], d['snr'].mean(), width=0.50, zorder=5, color=colors[i],
                        edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                        label=labels[i] if j == 0 else "", fill=True)

    ax0.legend(fontsize=9)
    ax0.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax0.set(ylabel='BER')
    ax0.set_yscale('log')
    ax0.set_xticks([])

    ax2.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax2.set(xlabel='Order of magnitude of the desired $BER_{max}$', ylabel='Energy consumption (J)')
    ax2.set_xticks(ranges)
    ax2.set_xticklabels(magnitudes)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax1.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax1.set_xticks([])
    ax1.set(ylabel='SNR')
    #ax1.set(xlabel='Order of magnitude of the desired $BER_{max}$', ylabel='SNR')
    #ax1.set_xticks(ranges)
    #ax1.set_xticklabels(magnitudes)
    #ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    #plt.savefig('results/s1_ber_energy.pdf', dpi=400, format='pdf')
    plt.show()


def plot_ber_energy():
    ranges = [0, 2, 4]
    magnitudes = [27, 54, 81]
    alpha = [0.6, 0.6]
    shift = [-0.25, 0]
    fill_pattern = ['', '']
    labels = ['DPLN', 'ADR']
    colors = ['tab:blue', 'tab:red']
    data = [data_ppo, data_ADR]

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 2])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    for i, d in enumerate(data):
        ber = d['ber']
        ber_th = d['ber_th']
        energy = d['energy_cons']

        size_range = 27

        for j, r in enumerate(ranges):
            inicio = j * size_range
            fin = (j + 1) * size_range
            ber = np.clip(ber, 1e-16, None)
            ber_mean = ber[inicio:fin].mean()
            ber_th_mean = ber_th[inicio:fin].mean()
            energy_mean = energy[inicio:fin].mean()

            ax0.bar(r + shift[i], ber_mean, width=0.25, zorder=5, color=colors[i],
                    edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels[i] if j == 0 else "", fill=True)

            ax1.bar(r + shift[i], energy_mean, width=0.25, zorder=5, color=colors[i],
                    edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels[i] if j == 0 else "", fill=True)
            if i == 0:
                ax0.bar(r + 0.25, ber_th_mean, width=0.25, zorder=5, color='none',
                        edgecolor='black', alpha=1, lw=1., label='$BER_{max}$' if j == 0 else "", fill=True)

    ax0.legend(fontsize=9, loc=(0.2, 0.6))
    ax0.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax0.set(ylabel='BER')
    ax0.set_yscale('log')
    ax0.set_xticks([])

    ax1.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax1.set(xlabel='Uplink messages', ylabel='Energy consumption (J)')
    ax1.set_xticks(ranges)
    ax1.set_xticklabels(magnitudes)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    plt.savefig('results/s1_ber_energy.pdf', dpi=400, format='pdf')
    plt.show()


def plot_comparative_config():
    df1 = data_ppo
    df2 = data_ADR
    pt_df1 = df1['pt']
    pt_df2 = df2['pt']
    sf_df1 = df1['sf']
    sf_df2 = df2['sf']

    # Crear la figura y los subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)

    # Graficar las variables en los subplots
    ax1.plot(sf_df1.index, sf_df1, color='tab:blue', label='DPLN', alpha=.7)
    ax1.plot(sf_df2.index, sf_df2,  color='tab:red', label='ADR', alpha=.7)
    ax1.set_ylabel('SF')
    ax1.grid(True, linestyle='-', alpha=0.4)
    ax1.legend(fontsize=9)

    ax2.plot(pt_df1.index, pt_df1, color='tab:blue', label='DPLN', alpha=.7)
    ax2.plot(pt_df2.index, pt_df2,  color='tab:red', label='ADR', alpha=.7)
    ax2.set_xlabel('Uplink messages')
    ax2.set_ylabel('TP (W)')
    ax2.grid(True, linestyle='-', alpha=0.4)
    ax2.legend(loc='center right', fontsize=9)

    plt.savefig('results/comparative.pdf', dpi=400, format='pdf')

    plt.show()


def plot_energy_nodes():
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.subplots(1, 1)

    nodes = [1, 3, 5]
    payload = [26, 78, 130]
    alpha = [0.6, 0.6, 0.6]
    shift = [-0.35, 0, 0.35]
    fill_pattern = ['', '//', '//']

    for n, nod in enumerate(nodes):
        data_ppo = pd.read_csv(local_dir + 'n_' + str(nod) + '_PPO_prueba.csv')
        data_SF7 = pd.read_csv(local_dir + 'n_' + str(nod) + '_SF7_prueba.csv')
        data_SF12 = pd.read_csv(local_dir + 'n_' + str(nod) + '_SF12_prueba.csv')

        data = [data_ppo, data_SF7, data_SF12]
        for j, d in enumerate(data):
            energy_value = MAX_BATTERY_LEVEL - d['e'].iloc[-1]
            if j >=1:
                ax1.bar(nod + shift[j], energy_value, width=0.35, zorder=5, color='none', edgecolor=colors[j],
                        alpha=alpha[j], label=labels[j] if n == 0 else "", hatch=fill_pattern[j], lw=1.)
            else:
                ax1.bar(nod + shift[j], energy_value, capsize=2, width=0.35, zorder=5, color=colors[j], alpha=alpha[j],
                        label=labels[j] if n == 0 else "")

    ax1.set_xticks(nodes)
    ax1.set_xticklabels(payload)
    #ax1.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax1.set_ylabel('Energy consumption (J)')
    ax1.set_xlabel('Payload (bytes)')
    ax1.legend(bbox_to_anchor=(0.5, 0.95), fontsize=8)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()

#plot_snr_comparison()
#plot_distance_comparison()
#plot_cdf(data_ppo['ber'], data_ADR['ber'])
#plot_comparative_config()
#plot_ber_energy()
#plot_energy_nodes()

#plot_ber_energy_magnitudes()
