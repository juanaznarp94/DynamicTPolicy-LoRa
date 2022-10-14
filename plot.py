# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
from matplotlib.pyplot import cm
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import random
import csv


# Constants used thorough the training env
# Data from https://saft4u.saftbatteries.com/en/iot/simulator/result/4919

# Real battery used: LSH 20
# Project parameters
#
#     Environment: Smart Agriculture & Environmental Preservation
#     Application: Air quality monitoring
#     Location: Outdoor
#     Geographic zone: Europe
#     Connectivity solution: LoRa
#     Data transmission: Every 10'
#     Life duration: 5 years
#     Cut-off Voltage: 2,2 V
#
# Consumption profile
#
#     Maximal peak current: 50 mA
#     Yearly consumption: 1,29 Ah
#     Total consumption: 6.45 Ah
#

T = 600  # seconds
BW = 125  # KHz
TDC = 1 / 100  # 1%
Q = 0.051  # Seconds
QMAX = 3600*TDC/Q
QR = T*TDC/Q
PACKET_SIZE = 26*8  # Bits
CAPACITY = 6.45  # Ah
VOLTAGE = 3.6  # V
CUT_OFF_VOLTAGE = 2.2  # V
MAX_BATTERY_LEVEL = CAPACITY * (VOLTAGE-CUT_OFF_VOLTAGE) * 3600  # J
ALL_ACTIONS = {
    "a1": {'CR': 4/5, 'SF': 7, 'alpha': -30.2580, 'beta': 0.2857, 'TXR': 3410, 'SNR': 0.0001778279},
    "a2": {'CR': 4/5, 'SF': 8, 'alpha': -77.1002, 'beta': 0.2993, 'TXR': 1841, 'SNR': 0.0000999999},
    "a3": {'CR': 4/5, 'SF': 9, 'alpha': -244.6424, 'beta': 0.3223, 'TXR': 1015, 'SNR': 0.0000562341},
    "a4": {'CR': 4/5, 'SF': 10, 'alpha': -725.9556, 'beta': 0.3340, 'TXR': 507, 'SNR': 0.0000316227},
    "a5": {'CR': 4/5, 'SF': 11, 'alpha': -2109.8064, 'beta': 0.3407, 'TXR': 253, 'SNR': 0.0000177827},
    "a6": {'CR': 4/5, 'SF': 12, 'alpha': -4452.3653, 'beta': 0.2217, 'TXR': 127, 'SNR': 0.0000099999},
    "a7": {'CR': 4/7, 'SF': 7, 'alpha': -105.1966, 'beta': 0.3746, 'TXR': 2663, 'SNR': 0.0001778279},
    "a8": {'CR': 4/7, 'SF': 8, 'alpha': -289.8133, 'beta': 0.3756, 'TXR': 1466, 'SNR': 0.0000999999},
    "a9": {'CR': 4/7, 'SF': 9, 'alpha': -1114.3312, 'beta': 0.3969, 'TXR': 816, 'SNR': 0.0000562341},
    "a10": {'CR': 4/7, 'SF': 10, 'alpha': -4285.4440, 'beta': 0.4116, 'TXR': 408, 'SNR': 0.0000316227},
    "a11": {'CR': 4/7, 'SF': 11, 'alpha': -20771.6945, 'beta': 0.4332, 'TXR': 204, 'SNR': 0.0000177827},
    "a12": {'CR': 4/7, 'SF': 12, 'alpha': -98658.1166, 'beta': 0.4485, 'TXR': 102, 'SNR': 0.0000099999}
}

def battery_life(action, N, MAX_BATTERY_LEVEL, T):
    config = list(ALL_ACTIONS.values())[action]
    cr = config.get("CR")
    sf = config.get("SF")
    payload = N*PACKET_SIZE/8  # bytes
    n_p = 8
    t_pr = (4.25 + n_p) * pow(2, sf) / BW
    p_sy = 8 + max(((8 * payload - 4 * sf + 44 - 20 * 1) / (4 * (sf - 2 * 1))) * (cr + 4), 0)
    t_pd = p_sy * pow(2, sf) / BW
    t = t_pr + t_pd
    e_pkt = 0.0924 * t  # J or Ws using Pt = 13 dBm
    battery_life = MAX_BATTERY_LEVEL * T / (e_pkt * 60 * 24 * 365)
    return battery_life

def plot_battery_life():
    N = np.linspace(1, 20, 40)
    labels = ['SF = 7', 'SF = 8', 'SF = 9', 'SF = 10', 'SF = 11', 'SF = 12']
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid()
    for a in range(round(len(ALL_ACTIONS) / 2)):
        y = []
        for i,n in enumerate(N):
            y.append(battery_life(a, n, MAX_BATTERY_LEVEL, T))
        ax1.plot(N, y, label=labels[a])
    ax1.set_ylabel('Battery life (Years)')
    ax1.set_xlabel('Number of external nodes (N)')
    ax1.set_xticks( [1, 5, 10, 15, 20])
    plt.suptitle('Battery life (T = 600 s, CR = 4/5)')
    ax1.legend(loc='best')
    plt.tight_layout()
    plt.savefig('battery_life.png', dpi=400)
    plt.show()

plot_battery_life()

