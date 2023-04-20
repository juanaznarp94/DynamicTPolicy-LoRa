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

Pt = 12 #dBm
Pt_max = 15
i = 0

for j in range(100): #quitar
    SNR_20_frames = []
    for i in range(20):
        SNR_20_frames.append(random.randint(-15, 15))
    SNR_max = np.max(SNR_20_frames)
    ALL_ACTIONS.values()[i]
    SNR_req = SNR_20_frames[0]

    SNR_margin = SNR_max - SNR_req - 10

    N_step = math.floor(SNR_margin/3)

    if N_step > 0:
        if ALL_ACTIONS.get('SF') > 7:
            i = i + 1
        else:
            Pt = Pt + 1
    if N_step < 0:
        if Pt != Pt_max:
            Pt = Pt + 1



