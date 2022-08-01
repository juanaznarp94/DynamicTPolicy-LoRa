import gym
from gym import spaces
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Ellipse
from matplotlib.collections import PatchCollection
import matplotlib.patches as matpatches
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

def plot_power():
    plt.figure(figsize=(8, 3))
    lambd = 5.08474576271 / 100 # Lambda (m)
    A = ((4*math.pi)/lambd)**2
    S = 6.309573444801942*1e-13 # Sensibilidad (W)
    m = 2
    beta = [2, 2.25, 2.5, 2.75, 3]
    color = ['b', 'r', 'g', 'm', 'orange']
    style = [':', '-.', '-', '-.', ':']
    powerw = np.linspace(0,30,30) # Ajustar (dBm)
    for i, b in enumerate(beta):
        powerd = []
        ranges = []
        for j, p in enumerate(powerw):
            pw = 10 ** (p / 10) / 1000
            rcs =  1.0 / (((S * A * m) / pw) ** (1.0 / b))
            powerd.append(p)
            ranges.append(rcs)
        plt.plot(ranges, powerd, label=r'$\beta$ = ' + str(b), lw=2, color = color[i], ls = style[i])
        plt.xlabel(r'$r_{CS}$ (m)')
        plt.ylabel('Tx Power (dBm)')

    plt.xlim(0,2000)
    plt.ylim(0,30)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.savefig('C:/...', dpi=300)
    plt.show()

plot_power()


