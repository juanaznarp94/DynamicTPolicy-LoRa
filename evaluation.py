# coding=utf-8
import csv
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
from loraEnv import loraEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, TD3, SAC
from sb3_contrib import RecurrentPPO

""" ----ENERGY----
num_nodes = [1, 5, 10]
sf_7 = [6.169430679111111e-06, 6.967902188444445e-06, 7.889215468444445e-06]
sf_12 = [1.9852852289777774e-05, 3.5576598935111115e-05, 5.719675057244444e-05]
"""

nodes = 1

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


def increasing_evaluation_ppo(local_dir):
    """
    This function simply evaluates the training environment using increasing BET_th and real distance measured over time
    :param local_dir:
    :return:
    """
    # EVALUATE AND SAVE RESULTS
    header = ['ber', 'ber_th', 'distance', 'distance_th', 'battery_life', 'energy_cons', 'prr', 'pdr', 'N', 'state',
              'algorithm']
    with open(local_dir+'main_results_ppo.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        #energy_ppo = 0
        #energy_sf7 = 0
        #energy_sf12 = 0
        for i, ber_th in enumerate(BER_TH_NORM):
            #state = env.set_ber_distance(BER_TH_NORM[0], REAL_DISTANCE_NORM[i], nodes)
            state = env.set_ber_distance(ber_th, REAL_DISTANCE_NORM[i], nodes)
            for k in range(100):
                action, _state = model_ppo.predict(state)  # predecimos la acción más recomendada para ese estado
                env.step(action)
                ber, ber_norm, distance, distance_norm, duration, c_total, prr, pdr, N, state = env.getStatistics()
                #energy_ppo = energy_ppo + c_total
                #energy_sf7 = energy_sf7 + sf_7[n]
                #energy_sf12 = energy_sf12 + sf_12[n]
                ber_th = BER_TH[int(np.where(BER_TH_NORM == ber_norm)[0])]
                distance_th = REAL_DISTANCE[int(np.where(REAL_DISTANCE_NORM == distance_norm)[0])]
                algorithm = 'PPO'
                data_row = [ber, ber_th, distance, distance_th, duration, c_total, prr, pdr, N, state, algorithm]
                writer.writerow(data_row)
        #return energy_ppo, energy_sf7, energy_sf12

def increasing_evaluation_a2c(local_dir):
    """
    This function simply evaluates the training environment using increasing BET_th and real distance measured over time
    :param local_dir:
    :return:
    """
    # EVALUATE AND SAVE RESULTS
    header = ['ber', 'ber_th', 'distance', 'distance_th', 'battery_life', 'energy_cons', 'prr', 'pdr', 'N', 'state',
              'algorithm']
    with open(local_dir+'main_results_a2c.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        #energy_a2c = 0
        #energy_sf7 = 0
        #energy_sf12 = 0
        for i, ber_th in enumerate(BER_TH_NORM):
            #state = env.set_ber_distance(BER_TH_NORM[0], REAL_DISTANCE_NORM[i], nodes)
            state = env.set_ber_distance(ber_th, REAL_DISTANCE_NORM[i], nodes)
            for k in range(100):
                action, _state = model_a2c.predict(state)  # predecimos la acción más recomendada para ese estado
                env.step(action)
                ber, ber_norm, distance, distance_norm, duration, c_total, prr, pdr, N, state = env.getStatistics()
                #energy_a2c = energy_a2c + c_total
                #energy_sf7 = energy_sf7 + sf_7[n]
                #energy_sf12 = energy_sf12 + sf_12[n]
                ber_th = BER_TH[int(np.where(BER_TH_NORM == ber_norm)[0])]
                distance_th = REAL_DISTANCE[int(np.where(REAL_DISTANCE_NORM == distance_norm)[0])]
                algorithm = 'A2C'
                data_row = [ber, ber_th, distance, distance_th, duration, c_total, prr, pdr, N, state, algorithm]
                writer.writerow(data_row)
        #return energy_a2c, energy_sf7, energy_sf12

def increasing_evaluation_rec(local_dir):
    """
    This function simply evaluates the training environment using increasing BET_th and real distance measured over time
    :param local_dir:
    :return:
    """
    # EVALUATE AND SAVE RESULTS
    header = ['ber', 'ber_th', 'distance', 'distance_th', 'battery_life', 'energy_cons', 'prr', 'pdr', 'N', 'state',
              'algorithm']
    with open(local_dir+'main_results_rec.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        #energy_rec = 0
        #energy_sf7 = 0
        #energy_sf12 = 0
        for i, ber_th in enumerate(BER_TH_NORM):
            #state = env.set_ber_distance(BER_TH_NORM[0], REAL_DISTANCE_NORM[i], nodes)
            state = env.set_ber_distance(ber_th, REAL_DISTANCE_NORM[i], nodes)
            for k in range(100):
                action, _state = model_rec.predict(state)  # predecimos la acción más recomendada para ese estado
                env.step(action)
                ber, ber_norm, distance, distance_norm, duration, c_total, prr, pdr, N, state = env.getStatistics()
                #energy_rec = energy_rec + c_total
                #energy_sf7 = energy_sf7 + sf_7[n]
                #energy_sf12 = energy_sf12 + sf_12[n]
                ber_th = BER_TH[int(np.where(BER_TH_NORM == ber_norm)[0])]
                distance_th = REAL_DISTANCE[int(np.where(REAL_DISTANCE_NORM == distance_norm)[0])]
                algorithm = 'PPO Recurrent'
                data_row = [ber, ber_th, distance, distance_th, duration, c_total, prr, pdr, N, state, algorithm]
                writer.writerow(data_row)
        #return energy_rec, energy_sf7, energy_sf12

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Load model
model_ppo = PPO.load("logs/best_model_ppo.zip")
model_a2c = A2C.load("logs/best_model_a2c.zip")
model_rec = RecurrentPPO.load("logs/best_model_rec.zip")  # para cargar modelo que se haya generado en el solver


#BER_TH = [0.00013895754823009532, 6.390550739301948e-05, 2.4369646975025416e-05, 7.522516546093483e-06,
#          1.8241669079988032e-06, 3.351781950877708e-07]

BER_TH = [4.2876832899021231e-05, 0.00014157493121903731, 9.562167102821912e-05, 6.952103521827612e-06,
          7.484709627069128e-07, 1.725129267129319e-06]

BER_TH_NORM = normalize_data(BER_TH)

REAL_DISTANCE = [4.047019336182087, 2.4728214673192119, 3.1245725782108312, 5.102794217312105, 8.00091792016258,
                 6.256321074217919]

#REAL_DISTANCE = [2.6179598590188147, 3.2739303314239954, 4.094264386099205, 5.12014586944165, 6.403077879720777,
#                 8.00746841578568]

REAL_DISTANCE_NORM = normalize_data(REAL_DISTANCE)

env = loraEnv(10)
state = env.reset()

# EVALUATE
##############################

local_dir = 'results/'
increasing_evaluation_ppo(local_dir)
increasing_evaluation_a2c(local_dir)
increasing_evaluation_rec(local_dir)

data_ppo = pd.read_csv(local_dir + 'main_results_ppo.csv')
data_a2c = pd.read_csv(local_dir + 'main_results_a2c.csv')
data_rec = pd.read_csv(local_dir + 'main_results_rec.csv')

# LINES -- BER AND DISTANCE

fig = plt.figure(figsize=(11, 6))
ax = fig.subplots(1, 2)

b_ppo = smooth(data_ppo['ber'], 18)
d_ppo = smooth(data_ppo['distance'], 18)
b_a2c = smooth(data_a2c['ber'], 18)
d_a2c = smooth(data_a2c['distance'], 18)
b_rec = smooth(data_rec['ber'], 18)
d_rec = smooth(data_rec['distance'], 18)

sns.lineplot(x=data_ppo.index, y=b_ppo, data=data_ppo, label="Moving average BER - PPO", alpha=.5, color='blue', ax=ax[0])
#sns.lineplot(x=data_ppo.index, y='ber', data=data_ppo, label="Measured BER - PPO", alpha=.1, color='blue', ax=ax[0])
sns.lineplot(x=data_a2c.index, y=b_a2c, data=data_a2c, label="Moving average BER - A2C", alpha=.5, color='green', ax=ax[0])
#sns.lineplot(x=data_a2c.index, y='ber', data=data_a2c, label="Measured BER - A2C", alpha=.1, color='green', ax=ax[0])
sns.lineplot(x=data_rec.index, y=b_rec, data=data_rec, label="Moving average BER - REC", alpha=.5, color='orange', ax=ax[0])
#sns.lineplot(x=data_rec.index, y='ber', data=data_rec, label="Measured BER - REC. PPO", alpha=.1, color='orange', ax=ax[0])
sns.lineplot(x=data_ppo.index, y='ber_th', data=data_ppo, label="Target BER", alpha=.5,
             color='red', ax=ax[0]).set(xlabel='Uplink messages', ylabel='BER')

ax[0].legend()

sns.lineplot(x=data_ppo.index, y=d_ppo, data=data_ppo, label="Moving average distance - PPO", alpha=.5, color='blue', ax=ax[1])
#sns.lineplot(x=data_ppo.index, y='distance', data=data_ppo, label="Estimated Maximum distance - PPO", alpha=.1, color='blue', ax=ax[1])
sns.lineplot(x=data_a2c.index, y=d_a2c, data=data_a2c, label="Moving average distance - A2C", alpha=.5, color='green', ax=ax[1])
#sns.lineplot(x=data_a2c.index, y='distance', data=data_a2c, label="Estimated Maximum distance - A2C", alpha=.1, color='green', ax=ax[1])
sns.lineplot(x=data_rec.index, y=d_rec, data=data_rec, label="Moving average distance - REC.", alpha=.5, color='orange', ax=ax[1])
#sns.lineplot(x=data_rec.index, y='distance', data=data_rec, label="Estimated Maximum distance - REC. PPO", alpha=.1, color='orange', ax=ax[1])
sns.lineplot(x=data_ppo.index, y='distance_th', data=data_ppo, label="Real distance", alpha=.5,
             color='red', ax=ax[1]).set(xlabel='Uplink messages', ylabel='Distance (Km)')
ax[1].legend()

fig.tight_layout()
#plt.figtext(0.5, 0.1, "Estimation of BER and distance for " + str(nodes) + "nodes", wrap=True, fontsize=10)
plt.savefig(local_dir+'lines.png', dpi=400)
plt.show()

"""
#ENERGY

en_ppo = []
en_ppo_7 = []
en_ppo_12 = []
en_a2c = []
en_a2c_7 = []
en_a2c_12 = []
en_rec = []
en_rec_7 = []
en_rec_12 = []

for n, nodes in enumerate(num_nodes):
    energy_ppo, energy_sf7, energy_sf12 = increasing_evaluation_ppo(local_dir)
    en_ppo.append(energy_ppo)
    en_ppo_7.append(energy_sf7)
    en_ppo_12.append(energy_sf12)
    #data_ppo = pd.read_csv(local_dir + 'main_results_ppo.csv')

    energy_a2c, energy_sf7, energy_sf12 = increasing_evaluation_a2c(local_dir)
    en_a2c.append(energy_a2c)
    en_a2c_7.append(energy_sf7)
    en_a2c_12.append(energy_sf12)
    #data_a2c = pd.read_csv(local_dir + 'main_results_a2c.csv')

    energy_rec, energy_sf7, energy_sf12 = increasing_evaluation_rec(local_dir)
    en_rec.append(energy_rec)
    en_rec_7.append(energy_sf7)
    en_rec_12.append(energy_sf12)
    #data_rec = pd.read_csv(local_dir + 'main_results_rec.csv')

fig = plt.figure(figsize=(7, 5))
shift = [-0.6, -0.3, 0, 0.3, 0.6]
# creating the bar plot
for i in range(len(en_ppo)):
    plt.bar(num_nodes[i] + shift[0], en_ppo_7[i], color='tab:brown', width=0.3, label='SF=7' if i == 0 else "")
    plt.bar(num_nodes[i] + shift[1], en_ppo[i], color='tab:blue', width=0.3, label='PPO' if i == 0 else "")
    plt.bar(num_nodes[i] + shift[2], en_a2c[i], color='tab:green', width=0.3, label='A2C' if i == 0 else "")
    plt.bar(num_nodes[i] + shift[3], en_rec[i], color='tab:red', width=0.3, label='RecurrentPPO' if i == 0 else "")
    plt.bar(num_nodes[i] + shift[4], en_ppo_12[i], color='tab:olive', width=0.3, label='SF=12' if i == 0 else "")

plt.xticks([1, 5, 10])
plt.xlabel("Number of nodes")
plt.ylabel("Energy (J)")
plt.title("Energy consumption")
leg = plt.legend()

for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
    legobj

plt.show()
plt.savefig('results/energy.png', dpi=400)

"""

"""
# BARS -- BER AND DISTANCE

fig = plt.figure(figsize=(9, 6))
ax = fig.subplots(1, 2)

sns.barplot(x='ber_th', y='ber', data=data,
            palette='deep', capsize=0.05, errwidth=1, ax=ax[0]).set(xlabel='Target BER', ylabel='BER')
for i, ber_th in enumerate(BER_TH[::-1]):
    ax[0].hlines(y=ber_th, xmin=i-0.5, xmax=0.5+i, color='red', alpha=0.3)
ax[0].set_xticklabels(["3.35e-07", "1.82e-06", "7.52e-06", "2.43e-05", "6.39e-05", "0.000138"], fontsize=8)
ax[0].set_title(str(nodes) + ' nodes')

sns.barplot(x='distance_th', y='distance', data=data,
            palette='deep', capsize=0.05, errwidth=1, ax=ax[1]).set(xlabel='Real distance', ylabel='distance')
for i, distance_th in enumerate(REAL_DISTANCE[::1]):
    ax[1].hlines(y=distance_th, xmin=i - 0.5, xmax=0.5 + i, color='red', alpha=0.3)
#ax[1].set_xticklabels(["2.62", "3.27", "4.09", "5.12", "6.4", "8.01"], fontsize=8)
ax[1].set_title(str(nodes) + ' nodes')
"""