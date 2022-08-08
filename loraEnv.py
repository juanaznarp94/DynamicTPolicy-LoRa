import gym
import math
import numpy as np
from abc import ABC
import numpy.random
from gym import spaces
from scipy import stats

# Constants used thorough the training env
T = 5  # seconds
AT = True  # if True, frontier node Always Transmit
BW = 125000  # Hz
TDC = 1 / 100  # 1%
Q = 0.051  # Seconds
QMAX = 3600*TDC/Q
QR = T*TDC/Q
PACKET_SIZE = 208  # Bits
MAX_BATTERY_LEVEL = 950  # mAh
MIN_BATTERY_RATIO = 0.8
ALL_ACTIONS = {
    "a1": {'CR': 4 / 5, 'SF': 7, 'alpha': -30.2580, 'beta': 0.2857, 'TXR': 3410, 'SNR': 0.0001778279},
    "a2": {'CR': 4 / 5, 'SF': 8, 'alpha': -77.1002, 'beta': 0.2993, 'TXR': 1841, 'SNR': 0.0000999999},
    "a3": {'CR': 4 / 5, 'SF': 9, 'alpha': -244.6424, 'beta': 0.3223, 'TXR': 1015, 'SNR': 0.0000562341},
    "a4": {'CR': 4 / 5, 'SF': 10, 'alpha': -725.9556, 'beta': 0.3340, 'TXR': 507, 'SNR': 0.0000316227},
    "a5": {'CR': 4 / 5, 'SF': 11, 'alpha': -2109.8064, 'beta': 0.3407, 'TXR': 253, 'SNR': 0.0000177827},
    "a6": {'CR': 4 / 5, 'SF': 12, 'alpha': -4452.3653, 'beta': 0.2217, 'TXR': 127, 'SNR': 0.0000099999},
    "a7": {'CR': 4 / 7, 'SF': 7, 'alpha': -105.1966, 'beta': 0.3746, 'TXR': 2663, 'SNR': 0.0001778279},
    "a8": {'CR': 4 / 7, 'SF': 8, 'alpha': -289.8133, 'beta': 0.3756, 'TXR': 1466, 'SNR': 0.0000999999},
    "a9": {'CR': 4 / 7, 'SF': 9, 'alpha': -1114.3312, 'beta': 0.3969, 'TXR': 816, 'SNR': 0.0000562341},
    "a10": {'CR': 4 / 7, 'SF': 10, 'alpha': -4285.4440, 'beta': 0.4116, 'TXR': 408, 'SNR': 0.0000316227},
    "a11": {'CR': 4 / 7, 'SF': 11, 'alpha': -20771.6945, 'beta': 0.4332, 'TXR': 204, 'SNR': 0.0000177827},
    "a12": {'CR': 4 / 7, 'SF': 12, 'alpha': -98658.1166, 'beta': 0.4485, 'TXR': 102, 'SNR': 0.0000099999}
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
    toremove = sum(to_transmit) - max_packets
    for g in [1, 2, 3]:  # Remove first low priorities, then high ones if still needed
        print(g)
        for i, v in enumerate(to_transmit_priorities):
            if toremove > 0:
                if v == g:
                    transmitted[i] = 0
                    toremove -= 1
    return transmitted


def H(a, b):
    """
    Heaviside function
    :param a: is 'a' overcomes 'b' return positive values, negative otherwise
    :param b: threshold
    :return: +-1
    """
    if a > b:
        return 1
    else:
        return -1


class loraEnv(gym.Env, ABC):
    """Lora Environment that follows gym interface"""

    def __init__(self, N):
        super(loraEnv, self).__init__()
        # Initialize variables
        self.action_space = spaces.Discrete(13)  # [0, 1, 2, 3, 4, ..., 12]
        self.min = 0
        self.max = max(QMAX, MAX_BATTERY_LEVEL)
        self.observation_space = spaces.Box(low=self.min, high=self.max, shape=(1,), dtype=np.float32)
        self.q = QMAX  # 706 at the very beginning
        self.g = 3
        self.e = MAX_BATTERY_LEVEL
        self.n = N
        self.tx = 1
        self.state = [self.q, self.g, self.e, self.n]
        self.packets_attempted = np.zeros((1, self.n))
        self.packets_transmitted = np.zeros((1, self.n))

    def step(self, action):
        """
        Given an action, decide whether transmission can occur or not (according to available energy and TDC constraint),
        and calculate metrics.
        :param action: an integer for each config
        :return: observation and reward
        """
        # Transform int "action" received to desired transmission config and extract variables (CR, SF, etc.)
        config = list(ALL_ACTIONS.values())[action]
        cr = config.get("CR")
        sf = config.get("SF")
        alpha = config.get("alpha")
        beta = config.get("beta")
        txr = config.get("TXR")
        snr = config.get("SNR")
        ########################################################################
        """ 
        CUIDADO: EL AGENTE SOLO APRENDE CUANDO Q=QMAX. 
        PARA FACILITAR EL APRENDIZAJE SE PUEDE ELIMINAR QR (EL TIEMPO EN QUE NO TRANSMITE), Y QUE AQUI SIEMPRE TENGAMOS Q=QMAX
        SIN EMBARGO, ESTO SI QUE SERIA BUENO METERLO EN LA EVALUACION PARA VER EL TIEMPO REAL
        """
        # Simulate transmissions of external nodes with Bernoulli processes and set random priorities.
        to_transmit = stats.bernoulli.rvs(0.5, size=self.n)  # !tx=0, tx=1
        print("--- to_transmit", to_transmit)
        priorities = np.random.randint(low=1, high=4, size=self.n, dtype=int)  # priority: low=1, medium=2, high=3
        to_transmit_priorities = np.multiply(to_transmit, priorities)
        print("--- to_transmit_priorities", to_transmit_priorities)

        # If Q equals QMAX (transmit only during 1% of T), there is energy enough,
        # and the frontier node has to transmit, that is to say, self.tx is 1,
        # then our packet is transmitted together with the packets received from external nodes
        if self.q == QMAX and self.e >= MIN_BATTERY_RATIO * MAX_BATTERY_LEVEL and self.tx == 1:
            max_packets = QMAX * txr * Q / PACKET_SIZE + 1
            print("--- max_packets", max_packets)
            if sum(to_transmit) > max_packets:
                transmitted = discard_lowest_g_packets(to_transmit, to_transmit_priorities, max_packets)
            else:
                transmitted = to_transmit

            # compute PDR
            self.packets_attempted = np.add(self.packets_attempted, to_transmit)
            self.packets_transmitted = np.add(self.packets_transmitted, transmitted)
            pdr = np.sum(self.packets_transmitted) / np.sum(self.packets_attempted)

            # compute PRR
            ber = pow(10, alpha * math.exp(beta * snr))
            prr = pow((1 - ber), PACKET_SIZE * sum(transmitted))
            """TASK: review SNR values, PRR is 1 -> too ideal, strange!"""

            # Update q value
            cost_action = sum(transmitted) * PACKET_SIZE / (txr * Q)
            self.q = max(self.q - cost_action, 0)  # update q value

            # compute Energy
            payload = PACKET_SIZE * sum(transmitted)
            h = 1
            de = 0
            n_p = 0
            n_payload = 8 + max((8 * payload - 4 * sf + 16 + 28 - 20 * h) / (cr - 2 * de), 0)
            t_symbol = pow(2, sf) / BW
            # t_preamble = (4.25 + n_p) * t_symbol
            # t_payload = n_payload * t_symbol
            p_cons = 412.5
            e_bit = (p_cons * (n_payload + n_p + 4.25) * t_symbol) / (8 * payload)
            self.e = self.e - e_bit * payload
            print("--- energy", self.e)
            """TASK: review units, energy is not working correctly"""

            # compute reward function
            reward = 10

        # STEP 3b) Cannot transmit! Update metrics
        else:
            # compute PDR
            self.packets_attempted = np.add(self.packets_attempted, to_transmit)
            # count attempted packets (this will worsen the final metric result)
            self.packets_transmitted = self.packets_transmitted
            # there is no variation in transmitted packets
            pdr = np.sum(self.packets_transmitted) / np.sum(self.packets_attempted)

            # No PRR can be computed if no packets are transmitted

            # Update q value
            self.q = min(self.q + QR, QMAX)  # update q value

            # compute Energy
            self.e = self.e

            # compute reward function
            reward = -1
            """TASK: define reward according to priorities of non transmitted packets"""

        info = {}
        done = False
        self.state = [self.q, self.g, self.e, self.n]
        observation = np.array(self.state)
        return observation, reward, done, info

    def reset(self):
        """Reset for training with fixed number of external nodes"""
        self.q = self.q
        if AT:
            self.g = 3
            self.tx = 1
        else:
            self.g = np.random.randint(1, 4)  # 1, 2, 3
            self.tx = np.random.randint(0, 2)  # 0, 1
        self.e = MAX_BATTERY_LEVEL
        self.n = self.n
        self.state = [self.q, self.g, self.e, self.n]
        observation = np.array(self.state)
        return observation

    def reset_variable_n(self):
        """Reset for learning variable number of external nodes"""
        self.q = QMAX
        if AT:
            self.g = 3
            self.tx = 1
        else:
            self.g = np.random.randint(1, 4)  # 1, 2, 3
            self.tx = np.random.randint(0, 2)  # 0, 1
        self.e = MAX_BATTERY_LEVEL
        self.n = np.random.randint(1, self.n)
        self.state = [self.q, self.g, self.e, self.n]
        observation = np.array(self.state)
        return observation

    def close(self):
        pass


env = loraEnv(5)

for i in range(2):
    env.step(1)
    print("Observation/State t = " + str(i) + ": " + str(env.state))
