import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, TD3, SAC
import math
import random

# An useful function to plot averages (typically used for rewards or whenever many iterations are plotted)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Load model
model = SAC.load("logs/best_model.zip")
# We can test, e.g., SAC and TD3.
# We need to check in the webpage of SB3 whether they are compatible with the kind of observation and action we are working with.
observation = []

# Evaluade model with predict() method
action = model.predict(observation)

# Load variables and run the frontier node n iterations while taking recommended actions.
# Collect data of interest and save in files

# It is very useful to save the data in some files, to later load them and plot in another script.
# This way you save a lot ot time while plotting, since you don't have to wait until the evaluation is done to plot.