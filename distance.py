import math

c = 3 * (10**8)  # speed of light (m/s)
f = 868 * (10**6)  # LoRa frequency Europe (Hz)
pt = 0.025  # Transmission power (W)
sf = [7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12]
snr_0 = 32  # value in lineal for SX1272 transceiver
nf = 4  # Noise figure in lineal (6 dB)
k = 1.380649 * (10**-23)  # Boltzmann constant (J/K)
t = 278  # temperature (K)
n = 3.1  # path loss exponent in urban area (2.7-3.5)
bw = [125000, 125000, 125000, 125000, 125000, 125000, 125000, 125000, 125000, 125000, 125000, 125000]  # Hz

for i in range(len(bw)):
    d = math.pow(math.pow(c/(4*math.pi*f), 2) * (pt * math.pow(2, sf[i])) / (snr_0*nf*k*t*bw[i]), 1/n)
    print('Distancia: ' + str(d * 0.001) + ' km')



