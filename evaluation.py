import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, TD3, SAC
import math
import random
from loraEnv import loraEnv

# An useful function to plot averages (typically used for rewards or whenever many iterations are plotted)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Load model
model = PPO.load("logs/best_model.zip") # para cargar modelo que se haya generado en el solver
# We can test, e.g., SAC and TD3.
# We need to check in the webpage of SB3 whether they are compatible with the kind of observation and action we are working with.
observation = []

prr = []
pdr = []
energy = []
ber = []
env = loraEnv(20)
#state = env.reset()
state = env.set_nodes(20)
count = 0

while env.e > 0:
    count += 1
    # Evaluade model with predict() method
    #action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
    env.step(5)
    state = env.state
    prr.append(env.get_prr())
    pdr.append(env.get_pdr())
    energy.append(env.get_energy())
    ber.append(env.get_ber())

print(count)
#battery_life = count * env.T / (60*24*365)  # cuanto tiempo (años) he estado tx antes de que se me acabe la batería
#print(battery_life)

np.savetxt("results/pdr_15_6.txt", pdr, delimiter=",")
np.savetxt("results/prr_15_6.txt", prr, delimiter=",")
np.savetxt("results/energy_15_6.txt", energy, delimiter=",")
np.savetxt("results/ber_15_6.txt", ber, delimiter=",")


# Load variables and run the frontier node n iterations while taking recommended actions.
# Collect data of interest and save in files

# It is very useful to save the data in some files, to later load them and plot in another script.
# This way you save a lot ot time while plotting, since you don't have to wait until the evaluation is done to plot.