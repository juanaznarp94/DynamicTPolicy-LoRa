import random
import gym
from gym import Env, spaces
import numpy.random
from gym import spaces
import numpy as np
import math
from scipy import special as sp


def normalize_value(value, arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_value = (value - arr_min) / (arr_max - arr_min)
    return normalized_value


# Constants used through the training env
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
MAX_BATTERY_LEVEL = CAPACITY * (VOLTAGE - CUT_OFF_VOLTAGE) * 3600
NODES = 10

BER = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

SNR = [0.01, 0.5, 1, 5, 10, 15, 20, 25, 30, 35]

ALLOWED_TPS = [0.012589, 0.025119, 0.1]

DISTANCE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

AVAILABLE_CONFIGS = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.])

DURATION_MAX = 32.178136874436944

DURATION_MIN = 0.16598592697738945


# Allowed actions (configurations)
ALL_ACTIONS = {
    "a1": {'DR': 0, 'CR': 4 / 5, 'SF': 12, 'SNR': -20, 'BW': 125, 'max_packages': math.floor(51 / PACKET_SIZE_BYTES)},
    "a2": {'DR': 1, 'CR': 4 / 5, 'SF': 11, 'SNR': -17.5, 'BW': 125, 'max_packages': math.floor(51 / PACKET_SIZE_BYTES)},
    "a3": {'DR': 2, 'CR': 4 / 5, 'SF': 10, 'SNR': -15, 'BW': 125, 'max_packages': math.floor(51 / PACKET_SIZE_BYTES)},
    "a4": {'DR': 3, 'CR': 4 / 5, 'SF': 9, 'SNR': -12.5, 'BW': 125, 'max_packages': math.floor(115 / PACKET_SIZE_BYTES)},
    "a5": {'DR': 4, 'CR': 4 / 5, 'SF': 8, 'SNR': -10, 'BW': 125, 'max_packages': math.floor(242 / PACKET_SIZE_BYTES)},
    "a6": {'DR': 5, 'CR': 4 / 5, 'SF': 7, 'SNR': -7.5, 'BW': 125, 'max_packages': math.floor(242 / PACKET_SIZE_BYTES)}
}

def distance(sf, bw, pt):
    bw = bw * 1000
    c = 3 * (10 ** 8)  # speed of light (m/s)
    f = 868 * (10 ** 6)  # LoRa frequency Europe (Hz)
    snr_0 = 32  # value in lineal for SX1272 transceiver
    nf = 4  # Noise figure in lineal (6 dB)
    k = 1.380649 * (10 ** -23)  # Boltzmann constant (J/K)
    t = 278  # temperature (K)
    n = 3.1  # path loss exponent in urban area (2.7-3.5)
    d = math.pow(math.pow(c / (4 * math.pi * f), 2) * (pt * math.pow(2, sf)) / (snr_0 * nf * k * t * bw), 1 / n) * 0.001
    return d


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


def heaviside(a, b):
    if b <= a:
        return 1
    else:
        return -1


def qfunc(x):
    return 0.5 - 0.5 * sp.erf(x / math.sqrt(2))


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


def time_on_air(payload_size: int, lora_param_sf, lora_param_cr, lora_param_crc, lora_param_bw, lora_param_h,
                lora_param_de):
    n_pr = 8  # https://www.google.com/patents/EP2763321A1?cl=en
    t_sym = (2.0 ** lora_param_sf) / lora_param_bw
    t_pr = (n_pr + 4.25) * t_sym
    payload_sym_n_b = 8 + max(
        math.ceil(
            (
                    8.0 * payload_size - 4.0 * lora_param_sf + 28 + 16 * lora_param_crc - 20 * lora_param_h) / (
                    4.0 * (lora_param_sf - 2 * lora_param_de)))
        * (lora_param_cr + 4), 0)
    t_payload = payload_sym_n_b * t_sym
    return t_pr + t_payload


def model_energy(payload, MAX_BATTERY_LEVEL, T, h, de, sf, cr, bw, pt):
    n_p = 8
    t_pr = (4.25 + n_p) * pow(2, sf) / bw
    p_sy = 8 + max(((8 * payload - 4 * sf + 44 - 20 * h) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    t_pd = p_sy * pow(2, sf) / bw
    t = t_pr + t_pd
    e_pkt = pt * t  # J or Ws using Pt = 13 dBm
    battery_life = MAX_BATTERY_LEVEL * T / (e_pkt * 60 * 24 * 365)
    return e_pkt, battery_life


class loraEnv(Env):
    """Lora Environment that follows gym interface"""

    def __init__(self, N):  # Method to initialize the attributes of the object we create
        super(loraEnv, self).__init__()

        # Class attributes
        self.action = 0
        self.sf = 7
        self.snr_measured = SNR[0]
        self.snr_measured_norm = normalize_value(self.snr_measured, SNR)
        self.ber_th = BER[0]
        self.ber_th_norm = normalize_value(self.ber_th, BER)
        self.distance_th = DISTANCE[0]
        self.distance_th_norm = normalize_value(self.distance_th, DISTANCE)
        self.pt = ALLOWED_TPS[0]
        self.pt_norm = normalize_value(self.pt, ALLOWED_TPS)
        self.q = Q_MAX
        self.e = CAPACITY
        self.n = N
        self.n_norm = N / NODES
        self.duration = 0
        self.action = 0
        self.i = 0
        self.t_tx = 0
        self.c_total = 0

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.MultiDiscrete([len(ALL_ACTIONS), len(ALLOWED_TPS)])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float64)
        #self.observation_space = spaces.MultiDiscrete([len(AVAILABLE_CONFIGS_NORM), len(BER_NORM), len(SNR_NORM),
        #                                               len(DISTANCE_NORM), NODES, len(ALLOWED_TPS_NORM)])
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th_norm, self.snr_measured_norm, self.distance_th_norm,
                      self.n_norm, self.pt_norm]

        self.pdr = 0
        self.prr = 0
        self.ber = 0
        self.snr = 0
        self.distance = 0

    def step(self, action):

        reward = -10  # by defect
        self.action = action
        # Transform action (int) in the desired config and create variables (CR, SF, alpha, etc)
        self.i = action[0]
        pt_w = action[1]
        config = list(ALL_ACTIONS.values())[self.i]
        self.pt = ALLOWED_TPS[pt_w]
        self.pt_norm = normalize_value(self.pt, ALLOWED_TPS)
        cr = config.get("CR")
        sf = config.get("SF")
        self.sf = sf
        bw = config.get('BW')
        snr = config.get('SNR')
        snr_lineal = 10**(snr/10)
        max_packages = config.get("max_packages")
        txr = sf * bw / (math.pow(2, sf))

        self.distance = distance(sf, bw, self.pt)

        to_transmit = np.ones(self.n).astype(int)
        priorities = np.random.randint(low=1, high=4, size=self.n, dtype=int)
        to_transmit_priorities = np.multiply(to_transmit, priorities)

        if sum(to_transmit) > max_packages:
            transmitted = discard_lowest_g_packets(to_transmit, to_transmit_priorities, max_packages)

        else:
            transmitted = to_transmit

        # PDR local
        self.pdr = np.sum(transmitted) / np.sum(to_transmit)

        # SNR
        self.snr = snr_lineal  # SNR config lineal

        # BER
        self.ber = 0.5 * qfunc(math.sqrt(2 * pow(2, sf) * self.snr_measured) - math.sqrt(1.386 * sf + 1.154))

        # PRR
        self.prr = (1 - self.ber) ** (PACKET_SIZE_BITS * sum(transmitted))

        # Update q value
        rest_action = ((PACKET_SIZE_BITS * sum(transmitted) / txr) / Q)
        self.q = self.q - rest_action

        # Energy Consumption
        h, de = h_de(sf, bw)
        crc = 1
        payload = np.sum(transmitted) * PACKET_SIZE_BYTES  # bytes
        #self.t_tx = time_on_air(int(payload), sf, cr, crc, bw, h, de) / 1000
        self.c_total, self.duration = model_energy(payload, MAX_BATTERY_LEVEL, T, h, de, sf, cr, bw, self.pt)
        self.e = self.e - self.c_total

        # Normalize values for reward (N=1)
        duration_max = DURATION_MAX
        duration_min = DURATION_MIN

        duration_norm = (self.duration - duration_min) / (duration_max - duration_min)

        # Calculate reward
        #reward = heaviside(self.snr_measured, self.snr) + heaviside(self.distance, self.distance_th) + heaviside(self.ber_th, self.ber) + duration_norm
        reward = duration_norm*heaviside(self.snr_measured, self.snr) + duration_norm*heaviside(self.distance, self.distance_th) + duration_norm*heaviside(self.ber_th, self.ber)

        # Update state
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th_norm, self.snr_measured_norm, self.distance_th_norm,
                      self.n_norm, self.pt_norm]
        observation = np.array(self.state)
        info = {}
        done = False
        return observation, reward, done, info

    def getStatistics(self):
        return [self.ber, self.ber_th, self.snr, self.snr_measured, self.distance, self.distance_th, self.duration,
                self.c_total, self.prr, self.pdr, self.n, self.sf, self.e, self.pt]

    def reset(self):
        self.q = Q_MAX  # 706 at the beginning
        self.e = CAPACITY
        self.n = random.randint(1, 10)
        self.n_norm = self.n/NODES
        self.pdr = 0
        self.prr = 0
        self.ber = 0
        self.snr = 0
        self.snr_measured = random.choice(SNR)
        self.snr_measured_norm = normalize_value(self.snr_measured, SNR)
        self.ber_th = random.choice(BER)
        self.ber_th_norm = normalize_value(self.ber_th, BER)
        self.i = np.random.randint(np.min(AVAILABLE_CONFIGS), len(AVAILABLE_CONFIGS))
        self.distance_th = random.choice(DISTANCE)
        self.distance_th_norm = normalize_value(self.distance_th, DISTANCE)
        self.pt = np.random.choice(ALLOWED_TPS)
        self.pt_norm = normalize_value(self.pt, ALLOWED_TPS)
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th_norm, self.snr_measured_norm, self.distance_th_norm,
                      self.n_norm, self.pt_norm]
        self.duration = 0
        observation = np.array(self.state)
        return observation

    def set_nodes(self, N):
        self.n = N
        self.n_norm = self.n/NODES
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th_norm, self.snr_measured_norm, self.distance_th_norm,
                      self.n_norm, self.pt_norm]
        observation = np.array(self.state)
        return observation

    def set_ber_distance_snr(self, ber_th, snr_measured, distance_th, N):
        self.distance_th = distance_th
        self.distance_th_norm = normalize_value(self.distance_th, DISTANCE)
        self.ber_th = ber_th
        self.ber_th_norm = normalize_value(self.ber_th, BER)
        self.snr_measured = snr_measured
        self.snr_measured_norm = normalize_value(self.snr_measured, SNR)
        self.n = N
        self.n_norm = N/NODES
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th_norm, self.snr_measured_norm, self.distance_th_norm,
                      self.n_norm, self.pt_norm]
        observation = np.array(self.state)
        return observation

    def set_ber(self, ber_th):
        self.ber_th = ber_th
        self.ber_th_norm = normalize_value(self.ber_th, BER)
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th_norm, self.snr_measured_norm, self.distance_th_norm,
                      self.n_norm, self.pt_norm]
        observation = np.array(self.state)
        return observation

    def set_ber_distance(self, ber_th, distance_th, N):
        self.ber_th = ber_th
        self.ber_th_norm = normalize_value(self.ber_th, BER)
        self.distance_th = distance_th
        self.distance_th_norm = normalize_value(self.distance_th, DISTANCE)
        self.n = N
        self.n_norm = self.n / NODES
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th_norm, self.snr_measured_norm, self.distance_th_norm,
                      self.n_norm, self.pt_norm]
        observation = np.array(self.state)
        return observation

    def set_distance(self, distance_th):
        self.distance_th = distance_th
        self.distance_th_norm = (self.distance_th - np.min(DISTANCE)) / (np.max(DISTANCE) - np.min(DISTANCE))
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th_norm, self.snr_measured_norm, self.distance_th_norm,
                      self.n_norm, self.pt_norm]
        observation = np.array(self.state)
        return observation

