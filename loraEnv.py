from gym import Env, spaces
import numpy.random
from gym import spaces
import numpy as np
import numpy.ma as ma
import math
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Ellipse, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as matpatches
import matplotlib.image as mpimg
from sympy import Point, Polygon
from shapely.geometry.polygon import LinearRing, Polygon
from math import dist
from scipy import stats
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

# Constants used through the training env
T = 600  # seconds
BW = 125  # KHz
TDC = 1 / 100  # 1%
Q = 0.051  # Seconds
Q_MAX = 3600*TDC/Q
QR = T*TDC/Q
PACKET_SIZE = 26*8  # Bits
CAPACITY = 6.45  # Ah
VOLTAGE = 3.6  # V
CUT_OFF_VOLTAGE = 2.2  # V
MAX_BATTERY_LEVEL = CAPACITY * (VOLTAGE-CUT_OFF_VOLTAGE) * 3600  # J

# Allowed actions (configurations)
ALL_ACTIONS = {
    "a1": {'CR': 4 / 5, 'SF': 7, 'alpha': -30.2580, 'beta': 0.2857, 'TXR': 3410, 'SNR': -7.5},
    "a2": {'CR': 4 / 5, 'SF': 8, 'alpha': -77.1002, 'beta': 0.2993, 'TXR': 1841, 'SNR': -10},
    "a3": {'CR': 4 / 5, 'SF': 9, 'alpha': -244.6424, 'beta': 0.3223, 'TXR': 1015, 'SNR': -12.5},
    "a4": {'CR': 4 / 5, 'SF': 10, 'alpha': -725.9556, 'beta': 0.3340, 'TXR': 507, 'SNR': -15},
    "a5": {'CR': 4 / 5, 'SF': 11, 'alpha': -2109.8064, 'beta': 0.3407, 'TXR': 253, 'SNR': -17.5},
    "a6": {'CR': 4 / 5, 'SF': 12, 'alpha': -4452.3653, 'beta': 0.2217, 'TXR': 127, 'SNR': -20},
    "a7": {'CR': 4 / 7, 'SF': 7, 'alpha': -105.1966, 'beta': 0.3746, 'TXR': 2663, 'SNR': -7.5},
    "a8": {'CR': 4 / 7, 'SF': 8, 'alpha': -289.8133, 'beta': 0.3756, 'TXR': 1466, 'SNR': -10},
    "a9": {'CR': 4 / 7, 'SF': 9, 'alpha': -1114.3312, 'beta': 0.3969, 'TXR': 816, 'SNR': -12.5},
    "a10": {'CR': 4 / 7, 'SF': 10, 'alpha': -4285.4440, 'beta': 0.4116, 'TXR': 408, 'SNR': -15},
    "a11": {'CR': 4 / 7, 'SF': 11, 'alpha': -20771.6945, 'beta': 0.4332, 'TXR': 204, 'SNR': -17.5},
    "a12": {'CR': 4 / 7, 'SF': 12, 'alpha': -98658.1166, 'beta': 0.4485, 'TXR': 102, 'SNR': -20}
}
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
    if a > b:
        return 1
    else:
        return -1


class loraEnv(Env):
    """Lora Environment that follows gym interface"""

    def __init__(self, N):  # Method to initialize the attributes of the object we create
        super(loraEnv, self).__init__()

        # Class attributes
        self.N = N
        self.min = 0
        self.max = max(max(Q_MAX, MAX_BATTERY_LEVEL), self.N)
        self.q = Q_MAX
        self.e = MAX_BATTERY_LEVEL
        self.n = self.N

        # Arrays to calculate pdr
        self.packets_attempted = np.zeros(self.n)  # to store the sum of the transmissions attempted by the external nodes
        self.packets_transmitted = np.zeros(self.n)  # to store the sum of the transmissions made by the external nodes

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(low=self.min, high=self.max, shape=(3,), dtype=np.float64)
        self.state = [self.q, self.e, self.n]

        self.pdr = -1
        self.prr = []
        self.ber = 0

    def step(self, action):
        # Called to take an action with the environment, it returns the next observation,

        reward = -1  # by defect

        # Transform action (int) in the desired config and create variables (CR, SF, alpha, etc)
        config = list(ALL_ACTIONS.values())[action]
        cr = config.get("CR")
        sf = config.get("SF")
        alpha = config.get("alpha")
        beta = config.get("beta")
        txr = config.get("TXR")
        snr = config.get("SNR")

        # Create an array with transmissions of external nodes with Bernoulli distribution
        # Create an array with random priorities
        to_transmit = np.ones(self.n).astype(int)
        print('To_transmit: ' + str(to_transmit))
        self.packets_attempted = (self.packets_attempted + to_transmit).astype(int)
        print('Packets_attempted: ' + str(self.packets_attempted))
        print('To_transmit: ' + str(to_transmit))

        priorities = np.random.randint(low=1, high=4, size=self.n, dtype=int)
        print('Priorities: ' + str(priorities))

        to_transmit_priorities = np.multiply(to_transmit, priorities)
        print('To_transmit_priorities: ' + str(to_transmit_priorities))

        # If Q = QMAX (transmit only during 1% of T), there is energy enough,
        # and the frontier node has to transmit, that is to say, self.tx is 1,
        # then out packet is transmitted together with the packets received from external nodes
        if MAX_BATTERY_LEVEL > 0:
            max_packages = self.n/2
            #max_packages = txr * Q * Q_MAX / PACKET_SIZE + 1  # Max number of packets agent node can transmit

            if sum(to_transmit) > max_packages:
                transmitted = discard_lowest_g_packets(to_transmit, to_transmit_priorities, max_packages)
            else:
                transmitted = to_transmit

            # PDR local
            self.pdr = np.sum(transmitted) / np.sum(to_transmit)
            print('PDR local: ' + str(self.pdr))

            """
            # PDR global
            self.packets_transmitted = np.add(self.packets_transmitted, transmitted)
            self.packets_attempted = np.add(self.packets_attempted, to_transmit)
            self.pdr = np.sum(self.packets_transmitted) / np.sum(self.packets_attempted)
            
            print('PDR global: ' + str(self.pdr))
            """

            # PRR
            self.ber = pow(10, alpha * math.exp(beta * snr))
            self.prr = (1 - self.ber) ** (PACKET_SIZE * sum(transmitted))
            print('PRR: ' + str(self.prr))

            # Update q value
            rest_action = ((PACKET_SIZE * sum(transmitted) / txr) / Q)
            self.q = self.q - rest_action

            # ENERGY
            payload = self.N * PACKET_SIZE / 8  # bytes
            n_p = 8
            t_pr = (4.25 + n_p) * pow(2, sf) / BW
            p_sy = 8 + max(((8 * payload - 4 * sf + 44 - 20 * 1) / (4 * (sf - 2 * 1))) * (cr + 4), 0)
            t_pd = p_sy * pow(2, sf) / BW
            t = t_pr + t_pd
            idle = 1.05833  # J
            rx = 0.0295488  # J
            sleep = 0.0300672  # J
            print('Self.e antes: ' + str(self.e))
            e_pkt = 0.0924 * t + idle + rx + sleep  # J or Ws using Pt = 13 dBm
            battery_life = MAX_BATTERY_LEVEL * T / (e_pkt * 60 * 24 * 365)
            self.e = self.e - e_pkt
            print('Self.e después: ' + str(self.e))

            print('Energía por paquete: ' + str(e_pkt) + ' J')
            print('Duración de la batería: ' + str(battery_life) + ' años')

            # reward
            reward = 0.4 * heaviside(self.prr, 0.8) + 0.4 * heaviside(self.pdr, 0.8) - 0.2 * heaviside(e_pkt, 437.8)
            print('Recompensa: ' + str(reward))

        # Not transmit
        else:
            # Calculate metrics
            # PDR
            self.packets_attempted = np.add(self.packets_attempted, to_transmit)
            self.packets_transmitted = self.packets_transmitted
            self.pdr = np.sum(self.packets_transmitted) / np.sum(self.packets_attempted)
            self.prr = 0

            # Update q value
            # self.q = self.q + QR

            # Energy
            self.e = self.e

            reward = -1

        # update state
        self.state = [self.q, self.e, self.n]
        observation = np.array(self.state)
        info = {}
        done = False
        return observation, reward, done, info

    def get_pdr(self):
        return self.pdr

    def get_prr(self):
        return self.prr

    def get_energy(self):
        return self.e

    def get_ber(self):
        return self.ber

    def reset(self):
        self.q = Q_MAX  # 706 at the beginning
        self.e = MAX_BATTERY_LEVEL
        self.n = np.random.randint(1, self.N)
        self.packets_attempted = np.zeros((1, self.n))
        self.packets_transmitted = np.zeros((1, self.n))
        self.pdr = -1
        self.prr = []
        self.state = [self.q, self.e, self.n]
        observation = np.array(self.state)
        return observation

    def set_nodes(self, N):
        self.q = Q_MAX  # 706 at the beginning
        self.e = MAX_BATTERY_LEVEL
        self.n = N
        self.state = [self.q, self.e, self.n]
        observation = np.array(self.state)
        return observation

