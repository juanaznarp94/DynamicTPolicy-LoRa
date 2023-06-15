import random

from gym import Env, spaces
import numpy.random
from gym import spaces
import numpy as np
import math
from scipy import special as sp
from scipy.stats import loguniform


T = 600  # seconds
TDC = 1 / 100  # 1%
Q = 0.051  # Seconds
Q_MAX = 3600 * TDC / Q
QR = T * TDC / Q
PACKET_SIZE_BITS = 26 * 8  # Bits
PACKET_SIZE_BYTES = 26
CAPACITY = 7.4  # Ah
VOLTAGE = 3  # V
CUT_OFF_VOLTAGE = 2.2  # V
MAX_BATTERY_LEVEL = CAPACITY * (VOLTAGE-CUT_OFF_VOLTAGE) * 3600  # J
NODES = 5
DURATION_MAX = 32.178136874436944
DURATION_MIN = 0.16598592697738945
c = 3 * (10**8)  # speed of light (m/s)
f = 868 * (10**6)  # LoRa frequency Europe (Hz)
snr_0 = 32  # value in lineal for SX1272 transceiver
nf = 4  # Noise figure in lineal (6 dB)
k = 1.380649 * (10**-23)  # Boltzmann constant (J/K)
t = 278  # temperature (K)
n = 3.1  # path loss exponent in urban area (2.7-3.5)


def normalize_value(value, arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_value = (value - arr_min) / (arr_max - arr_min)
    return normalized_value

def qfunc(x):
    return 0.5 - 0.5 * sp.erf(x / math.sqrt(2))


def heaviside(a, b):
    if b <= a:
        return 1
    else:
        return -1

def discard_lowest_g_packets(to_transmit, to_transmit_priorities, max_packets):
    """
    If the number of packets to transmit is higher than max allowed,
    then transmit only max allowed and calculate metrics (PDR, PRR, etc.)
    This is done by removing first those with the lowest priority.
    :param to_transmit: array of nodes that want to transmit, e.g., [1 0 1 1] => nodes with id 1, 3 and 4 transmit
    :param to_transmit_priorities: like to_transmit but including priorities (1,2,3). e.g., [2 0 3 1]
    :param max_packets: max number of packets according to configuration selected with action
    :return: transmitted messages, e.g., [0 0 1 1] => nodes with id 3 and 4 transmit
    """
    transmitted = np.copy(to_transmit)
    to_remove = sum(to_transmit) - max_packets
    for g in [1, 2, 3]:  # Remove first low priorities, then high ones if still needed
        for i, v in enumerate(to_transmit_priorities):
            if to_remove > 0:
                if v == g:
                    transmitted[i] = 0
                    to_remove -= 1
    return transmitted


def h_de(lora_param_sf, lora_param_bw):
    if lora_param_bw == 125 and lora_param_sf in [11, 12]:
        lora_param_de = 1
    else:
        lora_param_de = 0

    if lora_param_sf == 6:
        lora_param_h = 1
    else:
        lora_param_h = 0
    return lora_param_h, lora_param_de


def model_distance(pt, sf, bw):
    d = math.pow(math.pow(c / (4 * math.pi * f), 2) * (pt * math.pow(2, sf)) / (snr_0 * nf * k * t * (bw*1000)), 1 / n)
    return d * 0.001

def model_energy(payload, MAX_BATTERY_LEVEL, T, h, de, sf, cr, bw, pt):
    n_p = 8
    t_pr = (4.25 + n_p) * pow(2, sf) / bw
    p_sy = 8 + max(((8 * payload - 4 * sf + 44 - 20 * h) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    t_pd = p_sy * pow(2, sf) / bw
    t = t_pr + t_pd
    e_pkt = pt * t  # J or Ws using Pt = 13 dBm
    battery_life = MAX_BATTERY_LEVEL * T / (e_pkt * 60 * 24 * 365)
    return e_pkt, battery_life


ber_values = [-14, -4]
snr_values = [-19, 6]
distance_values = [0, 8]
range = [1e-14, 0.0001]
BER = loguniform.rvs(range[0], range[1])
SNR = random.uniform(-19, 6)
DISTANCE = round(random.uniform(0, 8), 2)
CONFIGS = np.array([0, 1/5, 2/5, 3/5, 4/5, 1])
ALLOWED_TPS = [0.012589, 0.025119, 0.1]

ALL_ACTIONS = {
    "a1": {'DR': 0, 'CR': 4 / 5, 'SF': 12, 'SNR': -20, 'BW': 125, 'max_packages': math.floor(51 / PACKET_SIZE_BYTES),
           'distance': 8.019743983718786},
    "a2": {'DR': 1, 'CR': 4 / 5, 'SF': 11, 'SNR': -17.5, 'BW': 125, 'max_packages': math.floor(51 / PACKET_SIZE_BYTES),
           'distance': 6.412893893148757},
    "a3": {'DR': 2, 'CR': 4 / 5, 'SF': 10, 'SNR': -15, 'BW': 125, 'max_packages': math.floor(51 / PACKET_SIZE_BYTES),
           'distance': 5.127995129055817},
    "a4": {'DR': 3, 'CR': 4 / 5, 'SF': 9, 'SNR': -12.5, 'BW': 125, 'max_packages': math.floor(115 / PACKET_SIZE_BYTES),
           'distance': 4.1005409541726525},
    "a5": {'DR': 4, 'CR': 4 / 5, 'SF': 8, 'SNR': -10, 'BW': 125, 'max_packages': math.floor(242 / PACKET_SIZE_BYTES),
           'distance': 3.278949315215729},
    "a6": {'DR': 5, 'CR': 4 / 5, 'SF': 7, 'SNR': -7.5, 'BW': 125, 'max_packages': math.floor(242 / PACKET_SIZE_BYTES),
           'distance': 2.62197323034004}
}


class loraEnv(Env):
    """Lora Environment that follows gym interface"""

    def __init__(self, N):  # Method to initialize the attributes of the object we create
        super(loraEnv, self).__init__()

        # Class attributes
        self.n = N
        self.ber_th = BER
        self.snr_measured = SNR
        self.distance_th = DISTANCE
        self.pt = random.choice(ALLOWED_TPS)
        self.action = (0, 0)
        self.i = np.random.randint(0, 5)
        self.t_tx = 0
        self.ber = 0
        self.snr = 0
        self.prr = 0
        self.sf = 0
        self.distance = 0
        self.e = MAX_BATTERY_LEVEL
        self.duration = 0
        self.c_total = 0
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.MultiDiscrete([len(ALL_ACTIONS), len(ALLOWED_TPS)])
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float64)
        self.state = [CONFIGS[self.i], normalize_value(np.log10(self.ber_th), ber_values),
                      normalize_value(self.snr_measured, snr_values), normalize_value(self.distance_th, distance_values),
                      normalize_value(self.pt, ALLOWED_TPS), self.n / NODES]

    def step(self, action):
        reward = -10
        self.i = action[0]
        pt_w = action[1]
        self.pt = ALLOWED_TPS[pt_w]
        config = list(ALL_ACTIONS.values())[self.i]
        cr = config.get("CR")
        sf = config.get("SF")
        self.sf = sf
        bw = config.get('BW')
        snr = config.get("SNR")
        distance = model_distance(self.pt, sf, bw)
        max_packages = config.get("max_packages")
        txr = sf * bw / (math.pow(2, sf))

        to_transmit = np.ones(self.n).astype(int)
        priorities = np.random.randint(low=1, high=4, size=self.n, dtype=int)
        to_transmit_priorities = np.multiply(to_transmit, priorities)

        if sum(to_transmit) > max_packages:
            transmitted = discard_lowest_g_packets(to_transmit, to_transmit_priorities, max_packages)

        else:
            transmitted = to_transmit

        # PDR local
        self.pdr = np.sum(transmitted) / np.sum(to_transmit)

        # BER
        self.ber = 0.5 * qfunc(math.sqrt(2 * pow(2, sf) * (10 ** (self.snr_measured / 10))) - math.sqrt(1.386 * sf + 1.154))

        # SNR
        self.snr = snr

        # PRR
        self.prr = (1 - self.ber) ** (PACKET_SIZE_BITS * sum(transmitted))

        #DISTANCE
        self.distance = distance

        # Energy Consumption
        h, de = h_de(sf, bw)
        payload = np.sum(transmitted) * PACKET_SIZE_BYTES  # bytes
        self.c_total, self.duration = model_energy(payload, MAX_BATTERY_LEVEL, T, h, de, sf, cr, bw, self.pt)
        self.e = self.e - self.c_total

        duration_max = DURATION_MAX
        duration_min = DURATION_MIN

        duration_norm = (self.duration - duration_min) / (duration_max - duration_min)

        reward = duration_norm*heaviside(self.snr_measured, self.snr) + duration_norm*heaviside(self.ber_th, self.ber) \
                 + duration_norm*heaviside(self.distance, self.distance_th)

        # Update state
        self.state = [CONFIGS[self.i], normalize_value(np.log10(self.ber_th), ber_values),
                      normalize_value(self.snr_measured, snr_values),
                      normalize_value(self.distance_th, distance_values), normalize_value(self.pt, ALLOWED_TPS),
                      self.n / NODES]
        observation = np.array(self.state)
        info = {}
        done = False
        return observation, reward, done, info


    def getStatistics(self):
        return [self.ber, self.ber_th, self.snr, self.snr_measured, self.distance, self.distance_th, self.duration,
                self.state, self.pt, self.e, self.n, self.ber - self.ber_th, self.distance - self.distance_th,
                self.c_total, self.sf]

    def reset(self):
        self.e = MAX_BATTERY_LEVEL
        self.n = random.randint(1, 5)
        self.duration = 0
        self.ber = 0
        self.snr = 0
        self.distance = 0
        self.ber_th = loguniform.rvs(range[0], range[1])
        self.snr_measured = random.uniform(-19, 6)
        self.distance_th = round(random.uniform(0, 8), 2)
        self.i = np.random.randint(0, 5)
        self.pt = random.choice(ALLOWED_TPS)
        self.state = [CONFIGS[self.i], normalize_value(np.log10(self.ber_th), ber_values),
                      normalize_value(self.snr_measured, snr_values),
                      normalize_value(self.distance_th, distance_values), normalize_value(self.pt, ALLOWED_TPS),
                      self.n / NODES]
        observation = np.array(self.state)
        return observation, self.i, self.pt

    def set_ber_snr_distance(self, ber_th, snr_measured, distance_th, N):
        self.ber_th = ber_th
        self.snr_measured = snr_measured
        self.distance_th = distance_th
        self.n = N
        self.state = [CONFIGS[self.i], normalize_value(np.log10(self.ber_th), ber_values),
                      normalize_value(self.snr_measured, snr_values),
                      normalize_value(self.distance_th, distance_values), normalize_value(self.pt, ALLOWED_TPS),
                      self.n / NODES]
        observation = np.array(self.state)
        return observation

    def set_ber_nodes(self, ber_th, N):
        self.ber_th = ber_th
        self.n = N
        self.state = [CONFIGS[self.i], normalize_value(np.log10(self.ber_th), ber_values),
                      normalize_value(self.snr_measured, snr_values),
                      normalize_value(self.distance_th, distance_values), normalize_value(self.pt, ALLOWED_TPS),
                      self.n / NODES]
        observation = np.array(self.state)
        return observation
