import math

c = 3 * (10**8)  # speed of light (m/s)
f = 868 * (10**6)  # LoRa frequency Europe (Hz)
pt = 0.025  # Transmission power (W)
sf = [7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12]
snr = [0.177827941, 0.1, 0.0562341325, 0.0316227766, 0.0177827941, 0.01, 0.177827941, 0.1, 0.0562341325, 0.0316227766,
       0.0177827941, 0.01]  # values in lineal
nf = 4  # Noise figure in lineal (6 dB)
k = 1.380649 * (10**-23)
t = 278  # K
n = 2.34  # path loss exponent
bw = [125000, 500000, 125000, 125000, 500000, 500000, 125000, 500000, 125000, 125000, 500000, 500000]  # Hz

for i in range(len(snr)):
    d = math.pow((math.pow(c/(4*math.pi*f), 2) * (pt*math.pow(2, sf[i])) / (snr[i] * nf * k * t * bw[i])), 1/2.34)
    print(d)
