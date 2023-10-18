import csv
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
from new import loraEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
import random
import math
from scipy import special as sp
from numpy import sin, cos, arccos, pi, round
import matplotlib.ticker as mticker


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

BER_TH = [1e-16,1e-15, 1e-14,1e-13, 1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05]

distances = []
for _ in range(88):
    random_number = round(random.uniform(6, 8), 2)
    distances.append(random_number)

local_dir = 'results/scenario2/'

env = loraEnv(1)

model_ppo = PPO.load("logs/best_model.zip")
model_a2c = A2C.load(os.path.join("logs/lora_rl_a2c_vuibk.zip"), env=env, custom_objects = {'observation_space': env.observation_space,
                                                                                     'action_space': env.action_space})

def evaluation_s2(local_dir, algorithm, model, ber_th):
    header = ['ber', 'ber_th', 'snr', 'snr_measured', 'distance', 'distance_th', 'duration', 'state', 'pt',
              'energy_cons', 'e', 'sf', 'nodes']
    snr_db_measured = 40
    #for n, nodes in enumerate(num_nodes):
    with open(local_dir + str(ber_th) + '_A2C_d6.csv', 'w', encoding='UTF8', newline='') as f:
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
                action, _state = model_ppo.predict(state)
                env.step(action)
                """
                if i <= 22:
                    ber_th = BER_TH[0]
                    nodes = 1
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    env.step(action)
                if i > 22 and i <= 62:
                    ber_th = BER_TH[1]
                    nodes = 1
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    env.step(action)
                if i > 62:
                    ber_th = BER_TH[2]
                    nodes = 1
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    """

            ber, ber_max, snr, snr_db, distance, distance_th, duration, state, pt_d, energy, N, ber_diff, \
            distance_diff, energy_cons, sf = env.getStatistics()
            data_row = [ber, ber_th, snr, snr_db_measured, distance, distance_th, duration, state, ALLOWED_TPS[pt],
                        energy_cons, energy, sf]
            writer.writerow(data_row)


def evaluation_s3(local_dir, algorithm, model):
    header = ['ber', 'ber_th', 'snr', 'snr_measured', 'distance', 'distance_th', 'duration', 'state', 'pt',
              'energy_cons', 'e', 'sf', 'nodes']
    snr_db_measured = 40
    #for n, nodes in enumerate(num_nodes):
    with open(local_dir + 'A2C_DPLN_s3.csv', 'w', encoding='UTF8', newline='') as f:
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
                if i <= 15:
                    ber_th = BER_TH[0]
                    nodes = 1
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    env.step(action)
                if i > 15 and i<= 22:
                    ber_th = BER_TH[0]
                    nodes = 3
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    env.step(action)
                if i > 22 and i <= 33:
                    ber_th = BER_TH[1]
                    nodes = 3
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    env.step(action)
                if i > 33 and i <= 51:
                    ber_th = BER_TH[1]
                    nodes = 2
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    env.step(action)
                if i > 51 and i <= 62:
                    ber_th = BER_TH[1]
                    nodes = 4
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    env.step(action)
                if i > 62 and i <= 64:
                    ber_th = BER_TH[2]
                    nodes = 4
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                if i > 64:
                    ber_th = BER_TH[2]
                    nodes = 1
                    state = env.set_ber_snr_distance(ber_th, snr_db_measured, dist, nodes)
                    action, _state = model_ppo.predict(state)
                    env.step(action)

            ber, ber_max, snr, snr_db, distance, distance_th, duration, state, pt_d, energy, N, ber_diff, \
            distance_diff, energy_cons, sf = env.getStatistics()
            data_row = [ber, ber_th, snr, snr_db_measured, distance, distance_th, duration, state, ALLOWED_TPS[pt],
                        energy_cons, energy, sf, nodes]
            writer.writerow(data_row)


def evaluate(ber):
    evaluation_s2(local_dir, "A2C", model_a2c, ber)
    #evaluation(local_dir, "A2C", model_a2c)
    #evaluation(local_dir, "RecurrentPPO", model_rec)
    #evaluation(local_dir, "SF7", model_ppo)
    #evaluation(local_dir, "SF12", model_ppo)


#for i, ber in enumerate(BER_TH):
#    evaluate(ber)

data_ppo = pd.read_csv(local_dir + 'DPLN.csv')
data_a2c = pd.read_csv(local_dir + 'A2C_DPLN.csv')
data_ADR = pd.read_csv(local_dir + 'ADR.csv')
#data_SF7 = pd.read_csv(local_dir + 'n_1_SF7.csv')
#data_SF12 = pd.read_csv(local_dir + 'n_1_SF12.csv')


data = [data_ppo, data_a2c, data_ADR]
colors = ['#89c1e6', '#fccc91', 'darkseagreen']
labels = ["PPO", "A2C", "ADR", "Min"]


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


def activation_ADR():
    header = ['ber', 'ber_th', 'snr', 'snr_measured', 'config', 'pt', 'energy_cons', 'sf']
    with open(local_dir + 'ADR.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        snr_db_measured = 46
        ber_th = BER_TH[0]
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
                ber_th = BER_TH[0]
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)

            #ACTIVATE ADR 1
            if j == 20:
                ber_th = BER_TH[0]
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
                ber_th = BER_TH[1]
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            #ACTIVATE ADR 2
            if j == 40:
                ber_th = BER_TH[1]
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
                ber_th = BER_TH[1]
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            #ACTIVATE ADR 3
            if j == 60:
                ber_th = BER_TH[1]
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
                ber_th = BER_TH[2]
                env.set_ber_snr_distance(ber_th, snr_db_measured, dist, 1)
                action = (i, index_Pt)
                env.step(action)
                Lpath = ((4 * math.pi * frec / c) ** 2) * ((dist * 1000) ** n_)
                snr_meas = Pt / (Lpath * nf * k * t * bw[0])
                snr_db_measured = 10 * math.log10(snr_meas) + 1
                SNR_TH.append(snr_db_measured)

            #ACTIVATE ADR 4
            if j == 80:
                ber_th = BER_TH[2]
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
                ber_th = BER_TH[2]
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
#activation_ADR()

def plot_snr_ber_comparison():
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(6, 6))
    colors = ['#89c1e6', '#fccc91']
    num = [0, 20, 40, 60, 80]

    for i, s in enumerate(data):
        if i >= 2:
            pass
        else:
            sns.lineplot(ax=ax0, x=s.index, y=s['snr'], data=s, label=labels[i], alpha=1, color=colors[i],
                         ls='-', linewidth=2)

    sns.lineplot(ax=ax0, x=data_ADR.index, y=data_ADR['snr'], data=data_ADR, label='ADR', alpha=1, color='darkseagreen',
                 ls='-', linewidth=2)
    sns.lineplot(ax=ax0, x=data_ppo.index, y=data_ppo['snr_measured'], data=data_ppo, label=r'$SNR_{measured}$',
                 alpha=1, color='#89c1e6', ls=':', linewidth=2)
    sns.lineplot(ax=ax0, x=data_a2c.index, y=data_a2c['snr_measured'], data=data_a2c, label=r'$SNR_{measured}$',
                 alpha=1, color='#fccc91', ls=':', linewidth=2)
    sns.lineplot(ax=ax0, x=data_ADR.index, y=data_ADR['snr_measured'], data=data_ADR, label=r'$SNR_{measured_{ADR}}$',
                 alpha=1, color='darkseagreen', ls=':', linewidth=2)

    legend_colors = ax0.legend(handles=[plt.Line2D([], [], color='#89c1e6', linestyle='-', linewidth=2),
                                        plt.Line2D([], [], color='#fccc91', linestyle='-', linewidth=2),
                                        plt.Line2D([], [], color='darkseagreen', linestyle='-', linewidth=2)],
                               labels=['PPO', 'A2C', 'ADR'], fontsize=10, loc=(0.8, 0.57))
    legend_styles = ax0.legend(handles=[plt.Line2D([], [], color='black', linestyle='-', linewidth=2),
                                        plt.Line2D([], [], color='black', linestyle=':', linewidth=2)],
                               labels=['$SNR_{min}$', 'SNR'], fontsize=10, loc=(0.765, 0.26))

    ax0.add_artist(legend_colors)
    ax0.add_artist(legend_styles)
    ax0.set_ylabel('SNR', fontsize=13)
    ax0.set_xticks([])
    ax0.grid(True, linestyle='-', alpha=0.4)

    for i, s in enumerate(data):
        ber = np.clip(s['ber'], 1e-19, None)
        if i < 2:
            sns.lineplot(ax=ax1, x=s.index, y=ber, data=s,
                         label=r'$BER_{measured}$', alpha=1, color=colors[i], ls=':', linewidth=2)
        else:
            sns.lineplot(ax=ax1, x=s.index, y=data_ppo['ber_th'], data=s, label=labels[i], alpha=1, color='#89c1e6',
                         ls='-', linewidth=2)
            sns.lineplot(ax=ax1, x=data_ADR.index, y=ber, data=data_ADR, label=r'$BER_{measured_{ADR}}$',
                        alpha=1, color='darkseagreen', ls=':', linewidth=2)

    sns.lineplot(ax=ax1, x=data_ADR.index, y=data_ADR['ber_th'], data=data_ADR, label='ADR', alpha=1, color='darkseagreen',
                 ls='-', linewidth=2)

    legend_styles = ax1.legend(handles=[plt.Line2D([], [], color='black', linestyle='-', linewidth=2),
                                        plt.Line2D([], [], color='black', linestyle=':', linewidth=2)],
                               labels=['$BER_{max}$', 'BER'], fontsize=10, loc='center left')

    #ax1.add_artist(legend_colors)
    ax1.add_artist(legend_styles)
    ax1.set_yscale('log')
    ax1.set_ylabel('BER', fontsize=13)
    ax1.set_xticks([])
    ax1.grid(True, linestyle='-', alpha=0.4)

    for i, d in enumerate(data):
        if i >= 2:
            pass
        else:
            sns.lineplot(ax=ax2, x=d.index, y=d['distance'], data=d,
                         label='Maximum distance ' + str(labels[i]) + ' $d_{max}$', alpha=.7, color=colors[i], linewidth=2)

            sns.lineplot(ax=ax2, x=data_ppo.index, y=data_ppo['distance_th'], data=data_ppo, label='Measured distance $d$' if i == 0 else "" ,
                        alpha=1, color=colors[i], linestyle=':', linewidth=2)

    sns.lineplot(ax=ax2, x=data_ADR.index, y=data_ADR['distance'], data=data_ADR, label='ADR', alpha=1,
                 color='darkseagreen', ls='-', linewidth=2)

    legend_styles = ax2.legend(handles=[plt.Line2D([], [], color='black', linestyle='-', linewidth=2),
                                        plt.Line2D([], [], color='black', linestyle=':', linewidth=2)],
                               labels=['Maximum distance $d_{max}$', 'Distance $d$'], fontsize=10, loc='upper left')
    #ax2.add_artist(legend_colors)
    ax2.add_artist(legend_styles)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax2.set_xticks(num)
    ax2.set_xlabel('Uplink messages', fontsize=13)
    ax2.set_ylabel('Distance (km)', fontsize=13)

    fig.tight_layout()
    plt.savefig('results/s2_snr_ber.pdf', dpi=400, format='pdf')
    plt.show()


def plot_ber_energy_magnitudes():
    ranges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    magnitudes = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5]
    alpha = [1, 1, 1]
    shift = [-0.35, 0, 0.35]
    fill_pattern = ['', '', '//']
    labels_BER = ['PPO', 'A2C', 'ADR']
    labels_SNR = ['PPO', 'A2C', 'ADR']
    labels_energy = ['PPO', 'A2C', 'ADR']
    colors = ['#89c1e6', '#fccc91', 'darkseagreen']

    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    for j, mag in enumerate(magnitudes):
        data_ppo = pd.read_csv(local_dir + '1e' + str(mag) + '_d2_.csv')
        data_a2c = pd.read_csv(local_dir + '1e' + str(mag) + '_A2C_d2.csv')
        data_ADR = pd.read_csv(local_dir + 'ADR_1e' + str(mag) + '_d2.csv')
        data = [data_ppo, data_a2c, data_ADR]

        for i, d in enumerate(data):
            filtered_ber = d['ber'][~np.isinf(d['ber'])]  # Filtrar valores "inf" en la columna 'ber'
            filtered_snr = d['snr_measured'][
                ~np.isinf(d['snr_measured'])]  # Filtrar valores "inf" en la columna 'snr_measured'
            filtered_energy = d['energy_cons'][
                ~np.isinf(d['energy_cons'])]  # Filtrar valores "inf" en la columna 'energy_cons'

            ax0.bar(ranges[j] + shift[i], filtered_ber.mean(),
                    yerr=np.std(filtered_ber, ddof=1) / np.sqrt(len(filtered_ber)),
                    error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), width=0.32, zorder=5,
                    color=colors[i], edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels_BER[i] if j == 0 else "", fill=True)

            ax1.bar(ranges[j] + shift[i], filtered_snr.mean(),
                    yerr=np.std(filtered_snr, ddof=1) / np.sqrt(len(filtered_snr)),
                    error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), width=0.32,
                    zorder=5, color=colors[i], edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels_SNR[i] if j == 0 else "", fill=True)

            ax2.bar(ranges[j] + shift[i], filtered_energy.mean(),
                    yerr=np.std(filtered_energy, ddof=1) / np.sqrt(len(filtered_energy)),
                    error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), width=0.32,
                    zorder=5, color=colors[i], edgecolor=colors[i], alpha=alpha[i], hatch=fill_pattern[i], lw=1.,
                    label=labels_energy[i] if j == 0 else "", fill=True)

        ax0.bar(ranges[j] + 0.71, data_ppo['ber_th'].mean(),
                yerr=np.std(data_ppo["ber_th"], ddof=1) / np.sqrt(len(data_ppo["ber_th"])),
                error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), width=0.32, zorder=5, color='#ef8582',
                edgecolor='#ef8582', alpha=1, lw=1., label='$BER_{max}$' if j == 0 else "", fill=True)
        ax1.bar(ranges[j] + 0.71, data_ppo['snr'].mean(),
                yerr=np.std(data_ppo["snr"], ddof=1) / np.sqrt(len(data_ppo["snr"])),
                error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), width=0.32, zorder=5, color='#ef8582',
                edgecolor='#ef8582', alpha=1, lw=1., label='SNR' if j == 0 else "", fill=True)

    ax0.legend(fontsize=10, ncol=5)
    ax0.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax0.set_ylabel('BER', fontsize=13)
    ax0.set_yscale('log')
    ax0.set_ylim([1e-18, 1e5])
    ax0.set_xticks([])

    ax1.legend(fontsize=10, ncol=5, loc='lower right')
    ax1.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax1.set_xticks([])
    ax1.set_ylabel('$SNR_{min}$', fontsize=13)
    ax1.set_ylim([-25, 15])

    ax2.legend(fontsize=10, ncol=5)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax2.set_xlabel('Order of magnitude of the desired $BER_{max}$', fontsize=13)
    ax2.set_ylabel('Energy \nconsumption (J)', fontsize=13)
    ax2.set_xticks(ranges)
    ax2.set_xticklabels(magnitudes)
    ax2.set_ylim([0, 38])

    fig.tight_layout()
    plt.savefig('results/s2_ber_energy_snr_d2.pdf', dpi=400, format='pdf')
    plt.show()


def plot_comparative_config():
    df1 = data_ppo
    df2 = data_a2c
    df3 = data_ADR
    pt_df1 = df1['pt']
    pt_df2 = df2['pt']
    pt_df3 = df3['pt']
    sf_df1 = df1['sf']
    sf_df2 = df2['sf']
    sf_df3 = df3['sf']
    labels_energy = ['PPO', 'A2C', 'ADR']

    # Crear la figura y los subplots
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(6, 6))

    # Graficar las variables en los subplots
    sns.lineplot(ax=ax0, x=sf_df1.index, y=sf_df1, data=sf_df1, color='#89c1e6', label='PPO', alpha=1, linewidth=2)
    sns.lineplot(ax=ax0, x=sf_df2.index, y=sf_df2, data=sf_df2, color='#fccc91', label='A2C', alpha=1, linewidth=2)
    sns.lineplot(ax=ax0, x=sf_df3.index, y=sf_df3, data=sf_df3, color='darkseagreen', label='ADR', alpha=1, linewidth=2)
    ax0.set_ylabel('SF', fontsize=13)
    ax0.grid(True, linestyle='-', alpha=0.4)
    ax0.legend(fontsize=11, loc='upper left')
    ax0.set_xticks([])

    sns.lineplot(ax=ax1, x=pt_df1.index, y=pt_df1, data=pt_df1, color='#89c1e6', alpha=1, linewidth=2)
    sns.lineplot(ax=ax1, x=pt_df2.index, y=pt_df2, data=pt_df2, color='#fccc91', alpha=1, linewidth=2)
    sns.lineplot(ax=ax1, x=pt_df3.index, y=pt_df3, data=pt_df3, color='darkseagreen', alpha=1, linewidth=2)
    ax1.set_ylabel('$p_t$ (W)', fontsize=13)
    ax1.grid(True, linestyle='-', alpha=0.4)
    ax1.set_xticks([])
    #ax1.legend(fontsize=11, loc='center right')

    for i, d in enumerate(data):
        battery_levels = []
        MAX_BATTERY_LEVEL = 21312
        for index, row in d.iterrows():
            energy_cons = row['energy_cons']
            MAX_BATTERY_LEVEL = MAX_BATTERY_LEVEL - energy_cons
            battery_levels.append(MAX_BATTERY_LEVEL)

        sns.lineplot(ax=ax2, x=d.index, y=battery_levels, data=d, alpha=1,
                     color=colors[i], ls='-', linewidth=2)

    #ax2.legend(fontsize=11)
    ax2.set_xticks
    ax2.set_xlabel('Uplink messages', fontsize=13)
    ax2.set_ylabel('Energy \nconsumption (J)', fontsize=13)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    fig.tight_layout()
    plt.savefig('results/s2_comparative.pdf', dpi=400, format='pdf')
    plt.show()


def plot_scenario3_payload():
    real_ranges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    labels_met = ['BER PPO', 'BER A2C', 'SNR PPO', 'SNR A2C']
    colors = ['#3e7ea8', '#d98e32', '#3e7ea8', '#d98e32']
    data_a2c = pd.read_csv(local_dir + 'A2C_DPLN_s3.csv')
    data_ppo = pd.read_csv(local_dir + 'DPLN_s3.csv')
    data = [data_ppo, data_a2c]

    fig = plt.figure(figsize=(7, 3))
    ax0 = fig.subplots(1, 1)
    ax1 = ax0.twinx()
    shift = [-2, 1]
    alpha = [1, 1, 1, 1]
    fill_pattern = ['', '', '//', '//']
    nodes = [1, 3, 2, 4, 1]
    colors_alg = ['green', 'blue', 'green', 'blue']
    colors_met = ['orange', 'green']

    size_range = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]

    bar_width = 1.5  # Ancho de las barras
    bar_spacing = 1.5  # Espaciado entre las barras

    for z, d in enumerate(data):
        for j, r in enumerate(size_range):
            if j < 9:
                start = size_range[j]
                end = size_range[j + 1]
                count_ber = 0
                count_snr = 0
                for i in range(start, end):
                    if d['ber'][i] > d['ber_th'][i]:
                        count_ber += 1
                    if d['snr_measured'][i] < d['snr'][i]:
                        count_snr += 1
                if count_snr == 0:
                    count_snr = 0.1
                if count_ber == 0:
                    count_ber = 0.1
                percentage_ber = count_ber / len(d) * 100
                print(percentage_ber)
                percentage_snr = count_snr / len(d) * 100
                print(percentage_snr)
                sns.lineplot(data=d['nodes'] * 26, x=d.index, y=d['nodes'] * 26, color='#ef8582', ax=ax0, linewidth=1,
                             zorder=0)

                ax1.bar(r + shift[z], percentage_snr, width=bar_width, zorder=6, color='none',
                        edgecolor=colors[z], alpha=alpha[z], hatch='//', lw=1.,
                        label=labels_met[z] if j == 0 else "", fill=False)

                ax1.bar(r + shift[z] + bar_spacing, percentage_ber, width=bar_width, zorder=6, color='none',
                        edgecolor=colors[z+2], alpha=alpha[z], hatch='', lw=1.,
                        label=labels_met[z + 2] if j == 0 else "", fill=False)

    #ax0.legend(fontsize=9, loc='upper left')
    ax1.legend(fontsize=9, loc='upper right')
    ax0.set_xticks(real_ranges)
    ax0.set_xlim(-5, 87)
    ax1.set_ylim([0, 10])
    ax0.yaxis.grid(True, linestyle='-', alpha=0.4)
    ax0.set(xlabel='Uplink messages')
    ax0.set_ylabel(ylabel='Payload (bytes)', color='#c7514e')
    ax1.set_ylabel(ylabel='% of BER and SNR infringement', color=colors[2])
    ax1.tick_params(axis='y', colors=colors[2])

    # Cambiar el color del eje y izquierdo a azul
    ax0.tick_params(axis='y', colors='#c7514e')

    # Formatear los valores del eje y como porcentajes sobre 100
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    plt.tight_layout()
    plt.savefig('results/s3_payload.pdf', dpi=400, format='pdf')
    plt.show()


plot_snr_ber_comparison()
plot_comparative_config()
plot_ber_energy_magnitudes()
plot_scenario3_payload()
