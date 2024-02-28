import math
import numpy as np
import random

ALL_ACTIONS = {
    "a1": {'CR': 4 / 5, 'SF': 12, 'RSSI': [-134, -131], 'BW': 125, 'DR': 0.293},
    "a2": {'CR': 4 / 5, 'SF': 12, 'RSSI': [-131, -129], 'BW': 250, 'DR': 0.585},
    "a3": {'CR': 4 / 5, 'SF': 10, 'RSSI': [-129, -128], 'BW': 125, 'DR': 0.976},
    "a4": {'CR': 4 / 5, 'SF': 12, 'RSSI': [-128, -126], 'BW': 500, 'DR': 1.718},
    "a5": {'CR': 4 / 5, 'SF': 10, 'RSSI': [-126, -125.5], 'BW': 250, 'DR': 1.953},
    "a6": {'CR': 4 / 5, 'SF': 11, 'RSSI': [-125.5, -123], 'BW': 500, 'DR': 2.148},
    "a7": {'CR': 4 / 5, 'SF': 9, 'RSSI': [-123, -120], 'BW': 250, 'DR': 3.515},
    "a8": {'CR': 4 / 5, 'SF': 9, 'RSSI': [-120, -117], 'BW': 500, 'DR': 7.031},
    "a9": {'CR': 4 / 5, 'SF': 8, 'RSSI': [-117, -114], 'BW': 500, 'DR': 12.50},
    "a10": {'CR': 4 / 5, 'SF': 7, 'RSSI': [-114], 'BW': 500, 'DR': 21.875}
}

ptx = 12 # dBm
pt_min = 5 # dBm
pt_max = 15 # dBm
sf_min = 7
sf_max = 12
i = 0
sf = ALL_ACTIONS.values()[i]
adr_margin = 10 # dBm

for j in range(100): # define iterations 
    SNR_20_frames = []
    SNR_20_frames.append(random.randint(-15, 15))
    if len(SNR_20_frames) >= 20:
        SNR_max = np.max(SNR_20_frames)
        SNR_margin = SNR_max - adr_snr_req[sf] - adr_margin
        N_step = math.floor(SNR_margin/3)

        if Nstep == 0:
            pass
        else:
            while Nstep > 0 and sf > sf_min:
                sf = sf - 1
                Nstep = Nstep - 1
            while Nstep > 0 and ptx > pt_min:
                ptx = ptx - 3
                Nstep = Nstep - 1
            while Nstep < 0 and ptx < pt_max:
                ptx = ptx + 3
                Nstep = Nstep + 1
            while Nstep < 0 and sf < sf_max:
                sf = sf + 1
                Nstep = Nstep + 1


