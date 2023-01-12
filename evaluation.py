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
#EVALUATE OPTIMAL

observation = []
prr = []
pdr = []
energy = []
ber = []
battery_life = []
packets_transmitted = []
#nodes = [1, 5, 10, 15, 20, 25, 30]
count = 60

env = loraEnv(1)
#state = env.set_nodes(30)
state = env.reset()

state = env.set_ber(7.522516546093483e-06)
while count >= 50:
    print(state)
    # Evaluade model with predict() method
    action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
    print('acción: ' + str(action))
    env.step(action)
    state = env.state
    prr.append(env.get_prr())
    pdr.append(env.get_pdr())
    energy.append(env.get_energy())
    ber.append(env.get_ber())
    battery_life.append(env.get_battery_life())
    #packets_transmitted.append(env.get_packets_tx())

    count = count - 1

print("Cambiamos de BER")
state = env.set_ber(1.8241669079988032e-06)
while count >= 40:
    print(state)
    # Evaluade model with predict() method
    action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
    print('acción: ' + str(action))
    env.step(action)
    state = env.state
    prr.append(env.get_prr())
    pdr.append(env.get_pdr())
    energy.append(env.get_energy())
    ber.append(env.get_ber())
    battery_life.append(env.get_battery_life())
    #packets_transmitted.append(env.get_packets_tx())

    count = count - 1

print("Cambiamos de BER")
state = env.set_ber(3.351781950877708e-07)
while count >= 30:
    print(state)
    # Evaluade model with predict() method
    action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
    print('acción: ' + str(action))
    env.step(action)
    state = env.state
    prr.append(env.get_prr())
    pdr.append(env.get_pdr())
    energy.append(env.get_energy())
    ber.append(env.get_ber())
    battery_life.append(env.get_battery_life())
    #packets_transmitted.append(env.get_packets_tx())

    count = count - 1

print("Cambiamos de BER")
state = env.set_ber(2.4369646975025416e-05)
while count >= 20:
    print(state)
    # Evaluade model with predict() method
    action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
    print('acción: ' + str(action))
    env.step(action)
    state = env.state
    prr.append(env.get_prr())
    pdr.append(env.get_pdr())
    energy.append(env.get_energy())
    ber.append(env.get_ber())
    battery_life.append(env.get_battery_life())
    #packets_transmitted.append(env.get_packets_tx())

    count = count - 1

print("Cambiamos de BER")
state = env.set_ber(6.390550739301948e-05)
while count >= 10:
    print(state)
    # Evaluade model with predict() method
    action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
    print('acción: ' + str(action))
    env.step(action)
    state = env.state
    prr.append(env.get_prr())
    pdr.append(env.get_pdr())
    energy.append(env.get_energy())
    ber.append(env.get_ber())
    battery_life.append(env.get_battery_life())
    #packets_transmitted.append(env.get_packets_tx())

    count = count - 1

print("Cambiamos de BER")
state = env.set_ber(0.00013895754823009532)
while count >= 0:

    # Evaluade model with predict() method
    action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
    print('acción: ' + str(action))
    env.step(action)
    state = env.state
    prr.append(env.get_prr())
    pdr.append(env.get_pdr())
    energy.append(env.get_energy())
    ber.append(env.get_ber())
    battery_life.append(env.get_battery_life())
    #packets_transmitted.append(env.get_packets_tx())

    # print(count)
    count = count - 1

np.savetxt("results/values_ber.txt", ber, delimiter=",")
"""
np.savetxt(f"results/prr_opt_{nodes}.txt", prr, delimiter=",")
np.savetxt(f"results/energy_opt_{nodes}.txt", energy, delimiter=",")
np.savetxt(f"results/ber_opt_{nodes}.txt", ber, delimiter=",")
np.savetxt(f"results/battery_life_opt_{nodes}.txt", battery_life, delimiter=",")
#np.savetxt(f"results/packets_txt_opt_{nodes}.txt", packets_transmitted, delimiter=",")

# EVALUATION FOR SPECIF ACTION 

observation = []
nodes = [1, 5, 10, 15, 20]
actions = [0, 1, 2, 3, 4, 5]

for n_nodes in nodes:
    for n_action in actions:
        prr = []
        pdr = []
        energy = []
        ber = []
        battery_life = []
        packets_transmitted = []
        env = loraEnv(n_nodes)
        #env = loraEnv(20)
        #state = env.reset()
        state = env.set_nodes(n_nodes)

        # Evaluade model with predict() method
        #action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
        env.step(n_action)
        energy.append(env.get_energy())
        pdr.append(env.get_pdr())
        prr.append(env.get_prr())
        ber.append(env.get_ber())
        packets_transmitted.append(env.get_packets_tx())
        battery_life.append(env.get_battery_life())

        np.savetxt(f"results/pdr_{n_nodes}_{n_action}.txt", pdr, delimiter=",")
        np.savetxt(f"results/packets_tx_{n_nodes}_{n_action}.txt", packets_transmitted, delimiter=",")
        np.savetxt(f"results/prr_{n_nodes}_{n_action}.txt", prr, delimiter=",")
        np.savetxt(f"results/ber_{n_nodes}_{n_action}.txt", ber, delimiter=",")
        np.savetxt(f"results/energy_{n_nodes}_{n_action}.txt", energy, delimiter=",")
        np.savetxt(f"results/battery_life_{n_nodes}_{n_action}.txt", battery_life, delimiter=",")
"""
"""
# EVALUATION FOR SPECIF ACTION 
prr = []
pdr = []
energy = []
ber = []
battery_life = []
packets_transmitted = []
env = loraEnv(30)
#env = loraEnv(20)
#state = env.reset()
state = env.set_nodes(30)
# Evaluade model with predict() method
#action, _state = model.predict(state)  # predecimos la acción más recomendada para ese estado
env.step(5)

prr.append(env.get_prr())
pdr.append(env.get_pdr())
ber.append(env.get_ber())
packets_transmitted.append(env.get_packets_tx())
battery_life.append(env.get_battery_life())

np.savetxt("results/pdr_30_5.txt", pdr, delimiter=",")
np.savetxt("results/prr_30_5.txt", prr, delimiter=",")
np.savetxt("results/ber_30_5.txt", ber, delimiter=",")
np.savetxt("results/packets_tx_30_5.txt", packets_transmitted, delimiter=",")
np.savetxt("results/battery_life_30_5.txt", battery_life, delimiter=",")
"""

# Load variables and run the frontier node n iterations while taking recommended actions.
# Collect data of interest and save in files

# It is very useful to save the data in some files, to later load them and plot in another script.
# This way you save a lot ot time while plotting, since you don't have to wait until the evaluation is done to plot.