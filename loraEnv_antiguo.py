import gym
import numpy.random
from gym import spaces
import numpy as np
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

TDC = 1 / 100
q = 0.051
QMAX = 3600 * TDC / q
Qr = 1 #preguntar
max_battery_level = 950
NUM_BITS = 208

class loraEnv(gym.Env):
    """Lora Environment that follows gym interface"""
    def heaviside(a, b):
        if a > b:
            return 1
        else:
            return -1

    def __init__(self, nodes):
        super(loraEnv, self).__init__()
        #self.N = len(nodes)
        #Numero d nodos q van a enviar al nodo frontera (N)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(13)
        #self.observation_space = spaces.Tuple((
          #  spaces.Discrete(self.N),
           # spaces.Discrete(self.N),
            #spaces.Discrete(self.N)))
        self.observation_space = spaces.Box(low=self.min, high=self.max, shape=(1,), dtype=np.float32)
        self.Q = 706
        self.G = 2
        self.E = max_battery_level
        self.N = 50
        self.state = [self.Q, self.G, self.E, self.N]

        # Definimos los diferentes parámetros de cada configuración
        #Incluir la acción de no transmitir
        a_0 =
        a_1 = {'CR': 4 / 5, 'SF': 7, 'ai': 'a1', 'alpha': -30.2580, 'beta': 0.2857, 'TXR': 3410, 'SNR': 0.0001778279,'NMax_Paq': self.N*3410*q/208}
        a_2 = {'CR': 4 / 5, 'SF': 8, 'ai': 'a2', 'alpha': -77.1002, 'beta': 0.2993, 'TXR': 1841, 'SNR': 0.0000999999,'NMax_Paq': self.N*1841*q/208}
        a_3 = {'CR': 4 / 5, 'SF': 9, 'ai': 'a3', 'alpha': -244.6424, 'beta': 0.3223, 'TXR': 1015, 'SNR': 0.0000562341,'NMax_Paq': self.N*1015*q/208}
        a_4 = {'CR': 4 / 5, 'SF': 10, 'ai': 'a4', 'alpha': -725.9556, 'beta': 0.3340, 'TXR': 507, 'SNR': 0.0000316227,'NMax_Paq': self.N*507*q/208}
        a_5 = {'CR': 4 / 5, 'SF': 11, 'ai': 'a5', 'alpha': -2109.8064, 'beta': 0.3407, 'TXR': 253, 'SNR': 0.0000177827,'NMax_Paq': self.N*253*q/208}
        a_6 = {'CR': 4 / 5, 'SF': 12, 'ai': 'a6', 'alpha': -4452.3653, 'beta': 0.2217, 'TXR': 127, 'SNR': 0.0000099999,'NMax_Paq': self.N*127*q/208}
        a_7 = {'CR': 4 / 7, 'SF': 7, 'ai': 'a7', 'alpha': -105.1966, 'beta': 0.3746, 'TXR': 2663, 'SNR': 0.0001778279,'NMax_Paq': self.N*2663*q/208}
        a_8 = {'CR': 4 / 7, 'SF': 8, 'ai': 'a8', 'alpha': -289.8133, 'beta': 0.3756, 'TXR': 1466, 'SNR': 0.0000999999,'NMax_Paq': self.N*1466*q/208}
        a_9 = {'CR': 4 / 7, 'SF': 9, 'ai': 'a9', 'alpha': -1114.3312, 'beta': 0.3969, 'TXR': 816, 'SNR': 0.0000562341,'NMax_Paq': self.N*816*q/208}
        a_10 = {'CR': 4 / 7, 'SF': 10, 'ai': 'a10', 'alpha': -4285.4440, 'beta': 0.4116, 'TXR': 408, 'SNR': 0.0000316227,'NMax_Paq': self.N*408*q/208}
        a_11 = {'CR': 4 / 7, 'SF': 11, 'ai': 'a11', 'alpha': -20771.6945, 'beta': 0.4332, 'TXR': 204, 'SNR': 0.0000177827,'NMax_Paq': self.N*204*q/208}
        a_12 = {'CR':4 / 7, 'SF': 12, 'ai': 'a12', 'alpha': -98658.1166, 'beta': 0.4485, 'TXR': 102, 'SNR': 0.0000099999,'NMax_Paq': self.N*102*q/208}
        # Defino array con las diferentes configuraciones
        self.actions = np.array[a_1, a_2, a_3, a_4, a_5, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12]
        # Defino arrays que voy a necesitar para calcular el pdr
        self.txt = np.zeros(self.N)  # Array donde voy a almacenar el sumatorio de las transmisiones realizadas de los nodos externos
        self.sg1 = np.zeros(self.N)  # Array donde voy a almacenar el sumatorio de las transmisiones intentadas por los nodos externos

        self.battery_capacity = 0.950  # Capacidad de la batería en mAh
        self.bw = 125000  # Ancho de banda en Hz
        print(2)

    def step(self, action):
        # Called to take an action with the environment, it returns the next observation,
        # the immediate reward, whether the episode is over and additional information
        reward = -1

        if self.Q == self.QMAX:
            tx = np.zeros(self.N)  # En este array definimos las transmisiones que se llevan a cabo de los nodos externos

            # Array 1xN para generar eventos Bernoulli
            p = 0.5
            bernoulli = stats.bernoulli(p)
            g1 = bernoulli.rvs(p, size=self.N)  # Ej: [1 0 0 0 1 1 0 1 1 1 0] (1's quieren tx 0's no)

            # Creamos otro array 1xN random entre 0,1 y 2 --> Prioridades
            g2 = np.random.randint(low=1, high=3, size=self.N, dtype=int) # Ej: [1 0 0 2 1 1 0 2 0 1 0]

            # Multiplicamos los arrays para dar prioridad a los vecinos
            G_N = np.dot(g1, g2) #Prioridad nodos externos. Ej: [1 0 0 0 1 1 0 2 0 1 0]

            config = self.actions[action] # Configuracion que elegimos que es la acción

            sg1 = np.add(self.sg1, g1) # Sumamos a sg1 g1, donde tengo los nodos que quieren transmitir en esa iteración para ir acumulando
            # Ej: [0 0 0 0 0 0 0 0 0 0 0] + [1 0 0 0 1 1 0 1 1 1 0] = [1 0 0 0 1 1 0 1 1 1 0]

            # Definimos el número de paquetes de los nodos externos que finalmente vamos a transmitir
            # num_paq es el número de paquetes que finalmente transmites otra y  máx paq el número máximo de paquetes que podrías transmitir
            if (config['NMax_Paq']) >= sum(g1):
                num_paq = sum(g1) # En caso de que el num máx de paq permitido sea >= al que quiero transmitir, transmito todos los que tengo en g1
                tx = g1  # En ese array guardo las transmisiones que finalmente se realizan de los nodos externos
                #En el caso del ej tendríamos esta opción y tx = [1 0 0 0 1 1 0 1 1 1 0]
            else:
                num_paq = config['NMax_Paq'] # En caso de que quieran tx más nodos que paq permitidos, se transmite el máximo posible y se elimina un paquete
                index = np.min(G_N[np.nonzero(G_N)]).index() # Aquí eliminamos el paquete con menor prioridad
                np.delete(g1, index)
                tx = g1

            txt = np.add(self.txt, tx) # Vamos sumando las transmisiones que finalmente se realizan
            # Ej: [0 0 0 0 0 0 0 0 0 0 0] + [1 0 0 0 1 1 0 1 1 1 0] = [1 0 0 0 1 1 0 1 1 1 0]

            pdr = np.divide(txt, sg1) # Calculamos el pdr diviendo los arrays txt y sg1
            # Ej: [1 0 0 0 1 1 0 1 1 1 0] / [1 0 0 0 1 1 0 1 1 1 0] = [1 0 0 0 1 1 0 1 1 1 0]
            # Hacemos la media d todos los elementos
            sumatorio = 0
            for i in pdr:
                sumatorio += pdr[i]
            pdr = sumatorio / len(pdr)  # 6/11 = 0.545454

            #else:
            #   raise ValueError("Received invalid action={} which is not part of the action space".format(action))

            done = False

            self.state = [self.Q, self.G, self.E, self.N]
            observation = np.array(self.state)

            self.Q = self.Q - ((208 * num_paq / config[5]) / q)

            ber = pow(10, config['alpha']*math.exp(config['beta']*config['SNR']))
            prr = (1 - ber) ** (208*num_paq)

            ## ENERGY !!
            if self.battery_capacity > 0.8 * max_battery_level:
                pl = 208*num_paq
                sf = config[1]
                h = 1
                de = 0
                n_p = 0 #lo incluimos en el payload
                n_payload = 8 + max((8 * pl - 4 * sf + 16 + 28 - 20 * h)/(config['SF'] - 2 * de), 0)
                t_symbol = pow(2, config['beta']) / self.bw
                t_preamble = (4.25 + n_p) * t_symbol
                t_payload = n_payload * t_symbol

                p_cons = 412.5

                e_bit = (p_cons * (n_payload + n_p + 4.25) * t_symbol) / (8 * pl)
                num_bits = 208 * num_paq

                self.battery_capacity = self.battery_capacity - e_bit * num_bits

                ## REWARD !!
                reward = 10 * self.heaviside(prr, 0.8) + 10 * self.heaviside(pdr, 0.8) - self.heaviside(e_bit*num_bits, 0.0042)

            else:
                reward = -1


            #reward - energia, prr, pdr, y prioridad
            info = {}
            return observation, reward, done, info

        else:
            self.Q = self.Q + self.Qr
            reward = -1

    def reset(self):
        # Called at the beginning of an episode, it returns an observation
        self.Q = self.Q
        self.G = np.random.randint(0, 2)
        self.E = np.random.rand(0)
        self.N = np.random.randint(1)
        self.state = [self.Q, self.G, self.E, self.N]
        observation = np.array(self.state)
        return observation  # reward, done, info can't be included

    def close(self):
        pass