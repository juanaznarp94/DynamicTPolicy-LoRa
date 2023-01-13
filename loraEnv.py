from gym import Env, spaces
import numpy.random
from gym import spaces
import numpy as np
import math
from scipy import special as sp


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Constants used through the training env
T = 600  # seconds
TDC = 1 / 100  # 1%
Q = 0.051  # Seconds
Q_MAX = 3600 * TDC / Q
QR = T * TDC / Q
PACKET_SIZE_BITS = 11 * 8  # Bits
PACKET_SIZE_BYTES = PACKET_SIZE_BITS / 8
CAPACITY = 13  # Ah
VOLTAGE = 3.6  # V
CUT_OFF_VOLTAGE = 2.2  # V
MAX_BATTERY_LEVEL = CAPACITY * (VOLTAGE - CUT_OFF_VOLTAGE)
BER = [0.00013895754823009532, 6.390550739301948e-05, 2.4369646975025416e-05, 7.522516546093483e-06,
       1.8241669079988032e-06, 3.351781950877708e-07]

BER_NORM = normalize_data(BER)

DISTANCE = [2.6179598590188147, 3.2739303314239954, 4.094264386099205, 5.12014586944165, 6.403077879720777,
            8.00746841578568]
DURATION = [15.635346635989748, 14.452773648384722, 12.968427828954564, 10.47161281273549, 7.560400710355283,
            4.858807480488823]

AVAILABLE_CONFIGS = np.array([0, 1/5, 2/5, 3/5, 4/5, 1])

# Allowed actions (configurations)
ALL_ACTIONS = {
    "a1": {'CR': 4 / 5, 'SF': 7, 'SNR': -7.5, 'BW': 125, 'SNR_lineal': 0.177827941,
           'max_packages': math.floor(222 / PACKET_SIZE_BYTES)},
    "a2": {'CR': 4 / 5, 'SF': 8, 'SNR': -10, 'BW': 125, 'SNR_lineal': 0.1,
           'max_packages': math.floor(222 / PACKET_SIZE_BYTES)},
    "a3": {'CR': 4 / 5, 'SF': 9, 'SNR': -12.5, 'BW': 125, 'SNR_lineal': 0.0562341325,
           'max_packages': math.floor(115 / PACKET_SIZE_BYTES)},
    "a4": {'CR': 4 / 5, 'SF': 10, 'SNR': -15, 'BW': 125, 'SNR_lineal': 0.0316227766,
           'max_packages': math.floor(51 / PACKET_SIZE_BYTES)},
    "a5": {'CR': 4 / 5, 'SF': 11, 'SNR': -17.5, 'BW': 125, 'SNR_lineal': 0.0177827941,
           'max_packages': math.floor(51 / PACKET_SIZE_BYTES)},
    "a6": {'CR': 4 / 5, 'SF': 12, 'SNR': -20, 'BW': 125, 'SNR_lineal': 0.01,
           'max_packages': math.floor(51 / PACKET_SIZE_BYTES)}
}
"""
    "a7": {'CR': 4 / 7, 'SF': 7, 'SNR': -7.5, 'BW': 125, 'SNR_lineal': 0.177827941,
           'max_packages': math.floor(222 / PACKET_SIZE_BYTES)},
    "a8": {'CR': 4 / 7, 'SF': 8, 'SNR': -10, 'BW': 125, 'SNR_lineal': 0.1,
           'max_packages': math.floor(222 / PACKET_SIZE_BYTES)},
    "a9": {'CR': 4 / 7, 'SF': 9, 'SNR': -12.5, 'BW': 125, 'SNR_lineal': 0.05623413251,
           'max_packages': math.floor(115 / PACKET_SIZE_BYTES)},
    "a10": {'CR': 4 / 7, 'SF': 10, 'SNR': -15, 'BW': 125, 'SNR_lineal': 0.0316227766,
            'max_packages': math.floor(51 / PACKET_SIZE_BYTES)},
    "a11": {'CR': 4 / 7, 'SF': 11, 'SNR': -17.5, 'BW': 125, 'SNR_lineal': 0.0177827941,
            'max_packages': math.floor(51 / PACKET_SIZE_BYTES)},
    "a12": {'CR': 4 / 7, 'SF': 12, 'SNR': -20, 'BW': 125, 'SNR_lineal': 0.01,
            'max_packages': math.floor(51 / PACKET_SIZE_BYTES)}
}
"""

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


def heaviside(ber_th, ber):
    if ber <= ber_th:
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


def model_energy(t_tx):

    # Indicate battery type here
    a_tx = 45 * 1e-3  # A
    t_rx = 0.54  # seconds
    a_rx = 15.2 * 1e-3  # A
    t_idle = 1.27  # seconds
    a_idle = 3 * 1e-3  # A
    t_sleep = T - (t_idle + t_rx + t_tx)  # seconds, calculate sleep time by substracting busy values to T
    a_sleep = 14 * 1e-6  # A

    c_tx = a_tx * t_tx / 3600  # Ah
    c_rx = a_rx * t_rx / 3600  # Ah
    c_idle = a_idle * t_idle / 3600  # Ah
    c_sleep = a_sleep * t_sleep / 3600  # Ah

    c_total = c_rx + c_tx + c_idle + c_sleep

    yearly = c_total * 6 * 24 * 365  # Ah yearly consumption
    # We send a packet 6 times per hour, and a year have 24*365 hours

    battery = 13 * 0.39  # calculate the fraction of 13.000 mAh used having into account cutoff and voltage values
    duration = battery / yearly  # divide the max battery available by the yearly amount, so we will obtain years

    return c_total, duration


class loraEnv(Env):
    """Lora Environment that follows gym interface"""

    def __init__(self, N):  # Method to initialize the attributes of the object we create
        super(loraEnv, self).__init__()

        # Class attributes
        self.N = N
        self.ber_th = BER_NORM[0]
        self.min = 0
        self.max = np.amax(AVAILABLE_CONFIGS)
        self.q = Q_MAX
        self.e = CAPACITY
        self.n = self.N
        self.duration = 0
        self.action = 0
        self.i = 0

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(3)
        # TODO Probar con multidiscret si no funciona. Aunque mejor Box.
        self.observation_space = spaces.Box(low=self.min, high=self.max, shape=(2,), dtype=np.float64)
        #self.observation_space = spaces.MultiDiscrete([len(AVAILABLE_CONFIGS), len(BER_NORM)])
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th]

        self.pdr = 0
        self.prr = 0
        self.ber = 0

    def step(self, action):

        reward = -10  # by defect
        self.action = action

        # Transform action (int) in the desired config and create variables (CR, SF, alpha, etc)
        if action == 0:
            if self.i == len(AVAILABLE_CONFIGS) - 1:
                pass
            else:
                self.i = self.i + 1
        if action == 1:
            pass
        if action == 2:
            if self.i == np.min(AVAILABLE_CONFIGS):
                pass
            else:
                self.i = self.i - 1

        config = list(ALL_ACTIONS.values())[self.i]
        cr = config.get("CR")
        sf = config.get("SF")
        bw = config.get('BW')
        snr_lineal = config.get("SNR_lineal")
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
        # self.ber = pow(10, alpha * math.exp(beta * snr))
        self.ber = 0.5 * qfunc(math.sqrt(2 * pow(2, sf) * snr_lineal) - math.sqrt(1.386 * sf + 1.154))

        # PRR
        self.prr = (1 - self.ber) ** (PACKET_SIZE_BITS * sum(transmitted))

        # Update q value
        rest_action = ((PACKET_SIZE_BITS * sum(transmitted) / txr) / Q)
        self.q = self.q - rest_action

        # Energy Consumption
        h, de = h_de(sf, bw)
        crc = 1
        payload = self.N * PACKET_SIZE_BYTES  # bytes
        t_tx = time_on_air(int(payload), sf, cr, crc, bw, h, de) / 1000
        c_total, self.duration = model_energy(t_tx)
        self.e = self.e - c_total

        # Normalize values for reward (N=1)
        ber_max = np.amax(BER)
        ber_min = np.amin(BER)
        duration_max = np.amax(DURATION)
        duration_min = np.amin(DURATION)

        duration_norm = (self.duration - duration_min) / (duration_max - duration_min)
        ber_norm = (self.ber - ber_min) / (ber_max - ber_min)

        # Calculate reward
        #reward = duration_norm * heaviside(self.ber_th, self.ber) * (1/self.ber)
        reward = duration_norm * heaviside(self.ber_th, ber_norm)

        # update state
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th]
        observation = np.array(self.state)
        info = {}
        done = False
        return observation, reward, done, info

    def getStatistics(self):
        return [self.ber, self.ber_th, self.duration, self.prr, self.state]

    def reset(self):
        self.q = Q_MAX  # 706 at the beginning
        self.e = CAPACITY
        self.n = 1
        self.pdr = 0
        self.prr = 0
        self.ber = 0
        self.ber_th = np.random.choice(BER_NORM)
        self.i = np.random.randint(np.min(AVAILABLE_CONFIGS), len(AVAILABLE_CONFIGS))
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th]
        self.duration = 0
        observation = np.array(self.state)
        return observation

    def set_nodes(self, N):
        self.n = N
        #self.state = [sf, snr_lineal, max_packages]
        #observation = np.array(self.state)
        #return observation

    def set_ber(self, ber_th):
        self.ber_th = ber_th
        self.state = [AVAILABLE_CONFIGS[self.i], self.ber_th]
        observation = np.array(self.state)
        return observation