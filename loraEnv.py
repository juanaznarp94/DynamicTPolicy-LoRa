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
Q = 0.051
QMAX = 3600 * TDC / Q
Qr = 5
max_battery_level = 950

class loraEnv(gym.Env):
    """Lora Environment that follows gym interface"""
    def heaviside(a, b):
        if a > b:
            return 1
        else:
            return -1


    def __init__(self, nodes, links):
        super(loraEnv, self).__init__()
        #self.N = len(nodes)
        #Numero d nodos q van a enviar al nodo frontera (N)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(1)
        #self.observation_space = spaces.Tuple((
          #  spaces.Discrete(self.N),
           # spaces.Discrete(self.N),
            #spaces.Discrete(self.N)))
        self.observation_space = spaces.Box(low=self.min, high=self.max, shape=(1,), dtype=np.float32)
        self.Q = np.zeros((1, 1))
        self.G = np.zeros((1, 1))
        self.E = np.zeros((1, 1))
        self.N = np.zeros((1, 1))
        self.state = [self.Q, self.G, self.E, self.N]
        #añadir comentario
        a_1 = [4 / 5, 7, 'a1', -30.2580, 0.2857, 3410, 0.0001778279,590]
        a_2 = [4 / 5, 8, 'a2', -77.1002, 0.2993, 1841, 0.0000999999,318]
        a_3 = [4 / 5, 9, 'a3', -244.6424, 0.3223, 1015, 0.0000562341,175]
        a_4 = [4 / 5, 10, 'a4', -725.9556, 0.3340, 507, 0.0000316227,87]
        a_5 = [4 / 5, 11, 'a5', -2109.8064, 0.3407, 253, 0.0000177827,43]
        a_6 = [4 / 5, 12, 'a6', -4452.3653, 0.2217, 127, 0.0000099999,21]
        a_7 = [4 / 7, 7, 'a7', -105.1966, 0.3746, 2663, 0.0001778279,460]
        a_8 = [4 / 7, 8, 'a8', -289.8133, 0.3756, 1466, 0.0000999999,253]
        a_9 = [4 / 7, 9, 'a9', -1114.3312, 0.3969, 816, 0.0000562341,141]
        a_10 = [4 / 7, 10, 'a10', -4285.4440, 0.4116, 408, 0.0000316227,70]
        a_11 = [4 / 7, 11, 'a11', -20771.6945, 0.4332, 204, 0.0000177827,35]
        a_12 = [4 / 7, 12, 'a12', -98658.1166, 0.4485, 102, 0.0000099999,17]
        self.actions = np.array([a_1, a_2, a_3, a_4, a_5, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12])
        self.txt = np.zeros(self.N)
        self.sg1 = np.zeros(self.N)

        self.battery_capacity = 950 #mAh 
        self.bw = 125 #kHz

    def step(self, action):
        # Called to take an action with the environment, it returns the next observation,
        # the immediate reward, whether the episode is over and additional information
        reward = -100

        if self.Q == self.QMAX:
            tx = np.zeros(self.N) #Añadir coment

            # Array 1xN para generar eventos Bernoulli
            p = 0.5
            bernoulli = stats.bernoulli(p)
            g1 = bernoulli.rvs(p, size=self.N)

            #Creamos otro array 1xN random entre 0,1 y 2 --> Prioridades
            g2 = np.random.randint(low=0, high=2, size=self.N, dtype=int)

            #Multiplicamos los arrays para dar prioridad a los vecinos
            G_N = np.dot(g1, g2) #Prioridad nodos externos

            config = self.actions[action] #configuracion q elegimos

            sg1 = np.add(self.sg1,g1)

            if (config[7]) >= sum(g1):
                num_paq = sum(g1) #Numero de paquetes de nodos ext
                tx = g1
            else:
                num_paq = config[7]
                index = np.min(G_N[np.nonzero(G_N)]).index()
                np.delete(g1, index)
                tx = g1

            txt = np.add(self.txt,tx)
            pdr = np.divide(txt,sg1)
            #hacer la media d todos los elementos

            #else:
            #   raise ValueError("Received invalid action={} which is not part of the action space".format(action))

            done = False

            self.state = [self.Q, self.G, self.E, self.N]
            observation = np.array(self.state)

            self.Q = self.Q - ((208 * num_paq / config[5]) / self.Q)

            ber = pow(10, config[3]*math.exp(config[4]*config[6]))
            prr = (1 - ber) ** (208*num_paq)

            ## ENERGY !!
            if self.battery_capacity > 0.8*max_battery_level:
                pl = 208*num_paq
                sf = config[1]
                h = 1
                de = 0
                n_p = 0 #lo incluimos en el payload
                n_payload = 8 + max((8*pl - 4* sf + 16 +28 -20*h)/(config[1] - 2*de),0)
                t_symbol = pow(2, config[4])/ self.bw
                t_preamble = (4.25 + n_p) * t_symbol
                t_payload = n_payload * t_symbol

                p_cons = 412.5

                e_bit = (p_cons * (n_payload + n_p + 4.25) * t_symbol) / (8*pl)
                num_bits = 208*num_paq

                self.battery_capacity = self.battery_capacity - e_bit * num_bits

                ## REWARD !!
                reward = 10*self.heaviside(prr,0.8) + 10*self.heaviside(pdr,0.8) - self.heaviside(e_bit*num_bits,)

            else:
                reward = -400


            #reward - energia, prr, pdr, y prioridad
            info = {}
            return observation, reward, done, info

        else:
            self.Q = self.Q + self.Qr
            reward = -500

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


gw = (0,0) # Gateway o Base Station
p0 = (1000, 2000) # Nodo 0
p1 = (2000, 2000) # Nodo 1
p2 = (2000, 1000) # ...
p3 = (1500, 3000)
p4 = (3500, 1500)
p5 = (3000, 2500)
p6 = (3500, 4000)
p7 = (2000, 4000)

nodes = [
    gw,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7
]

buildings = [
    Polygon([(2500, 500), (2500, 2000), (3000, 2000), (3000, 500)]),
    Polygon([(2500, 3000), (2500, 3500), (3000, 3500), (3000, 3000)]),
    Polygon([(4000, 2000), (4000, 2500), (4500, 2500), (4500, 1000), (4250, 1000), (4250, 2000)]),
    Polygon([(500, 2300), (500, 4000), (1300, 4000), (1300, 2300), (500, 2300)]),
    Polygon([(1700, 4500), (1700, 5000), (4500, 5000), (4500, 4500)])
]

"""
TAREA 0: 
Pasar todo a Km (Opcional)
"""
#nodes_km = []
#for i, n in enumerate(nodes):
#    nodes_km[i] = nodes[i] /

#buildings_km = []

"""
TAREA 1: 
Implementar función compute_links()
"""
#dados los nodos y los edificios calcular si entre cada par de nodos hay un enlace o no
#rcs = math.gamma(m + 1.0 / beta) / (math.gamma(m) * (S * A * m / p) ** (1.0 / beta))
    # Si dos nodos NO tienen ningún edificio en medio -> (m = 2, beta = 2), calcular rcs y ver si están en rango o no
    # Si dos nodos SÍ tienen algún edificio en medio -> (m = 4, beta = 4), calcular rcs y ver si están en rango o no
    # - si no están en rango links[i][j] <-- 0
    # - si sí están en rango links[i][j] <-- probabilidad p aleatoria 0-0.5
    #links = [] # Matriz N*N, p.e., 1 si existe LOS y se garantiza recepción, 0 si no, 0.3 si existe interferencia

def compute_links(nodes, buildings):
    links = [[], []]
    #for i,n in enumerate(nodes):
     #   links.append([0] * 8)
    LAMBDA = 1
    A = ((4 * math.pi) / LAMBDA) ** 2
    S_dbm = -123
    p_dbm = 14
    p = 10 ** (p_dbm / 10) / 1000 #En W
    S = 10 ** (S_dbm / 10) / 1000 #En W
    b = 0
    for i, n in enumerate(nodes):
        for j, o in enumerate(nodes):
            poly1 = Polygon([nodes[i], nodes[j], nodes[j]])
            for z, u in enumerate(buildings):
                poly2 = buildings[z]
                isIntersection = poly1.intersection(poly2)  # si la linea recta entre el nodo y el gw intersectan con el edificio
                if isIntersection:
                    b = b+1
                    m = 4
                    beta = 4
                    rcs = math.gamma(m + 1.0 / beta) / (math.gamma(m) * (S * A * m / p) ** (1.0 / beta))
                    d = dist(nodes[i], nodes[j])
                    if rcs < d:
                        print(i, j)
                        links[i][j] = 0
                    else:
                        links[i][j] = 0.3
                    break
            if b == 0:
                m = 2
                beta = 2
                d = dist(nodes[i], nodes[j])
                rcs = math.gamma(m + 1.0 / beta) / (math.gamma(m) * (S * A * m / p) ** (1.0 / beta))
                if rcs < d:
                    links[i][j] = 0
                else:
                    links[i][j] = 1
    return links

"""
TAREA 2: 
Añadir links como parámetro de entrada a función plot_map() y representar enlaces con líneas entre dos puntos o nodos
"""
def plot_map(nodes, buildings, links):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    # Plot nodes including GW
    for i,n in enumerate(nodes):
        if i==0:
            # Plot GW
            ax1.scatter(50, 50, s=300, zorder=2, color='k', marker='^')
            ax1.annotate("GW", (100, 100))
            circle = Circle((0, 0), 3230, ls="--", alpha=1, fill=False, edgecolor='g', lw=1)
            ax1.add_patch(circle)
        else:
            # Plot LoRa nodes
            ax1.scatter(n[0], n[1], s=25, zorder=2, color='k', marker='s')
            ax1.annotate("ID: "+str(i), (n[0]+50,n[1]+50), color='k', fontsize='small')

    # Plot interfering buildings
    for i,b in enumerate(buildings):
        x,y = b.exterior.xy
        ax1.fill_between(x,y, alpha=0.2, hatch="x", color='r', edgecolor=None, linewidth=0.0)
        ax1.annotate("ID: " + str(i), (x[0]+50, y[0]+50), color='r', fontsize='small')

    # Plot links
    for i in nodes:
        for j in nodes:
            if links[i][j] > 0:
                plt.scatter(nodes[i], nodes[j])
                plt.plot(nodes[i], nodes[j])

    plt.grid(True)
    plt.ylabel("pos Y (m)")
    plt.xlabel("pos X (m)")
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)
    plt.savefig('map.png', dpi=300)
    plt.show()

"""
TAREA 3:
Implementar la función compute_paths()
"""
def compute_paths(nodes, links):
    paths = []
    #


    #
    return paths

compute_links(nodes, buildings)
plot_map(nodes, buildings, links)