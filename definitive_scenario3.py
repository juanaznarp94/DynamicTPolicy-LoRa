import csv
import math
import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
from loraEnv import loraEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, TD3, SAC
from sb3_contrib import RecurrentPPO
from gps_class import GPSVis
from numpy import sin, cos, arccos, pi, round
import cv2

# INIT
latitud_gw = 47.66577
longitud_gw = -122.34737
num_nodes = [1, 5, 10]


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees


def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians


def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2, unit='kilometers'):
    theta = longitude1 - longitude2

    distance = 60 * 1.1515 * rad2deg(
        arccos(
            (sin(deg2rad(latitude1)) * sin(deg2rad(latitude2))) +
            (cos(deg2rad(latitude1)) * cos(deg2rad(latitude2)) * cos(deg2rad(theta)))
        )
    )

    if unit == 'miles':
        return round(distance, 2)
    if unit == 'kilometers':
        return round(distance * 1.609344, 2)


def combine_csv_directory(local_dir):
    """
    Not used anymore
    :param local_dir:
    :return:
    """
    files = os.path.join(local_dir, "*.csv")
    files = glob.glob(files)
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.to_csv(local_dir + "total.csv")
    return df


df = pd.read_csv("trajectory_gps.csv", sep=',', header=0)  # load dataset
data = df[15:102]  # choosing a trajectory of 8 km
vis = GPSVis(data_path=data,
             map_path='map.png',  # Path to map downloaded from the OSM.
             points=(47.6763, -122.3839, 47.5902, -122.2634))
vis.create_image(color=(0, 0, 255), width=3)  # Set the color and the width of the GNSS tracks.
vis.plot_map(output='save')

distances = []

for i in range(len(data)):
    longitud = data.iloc[i]['Longitude']
    latitud = data.iloc[i]['Latitude']
    dist = getDistanceBetweenPointsNew(latitud_gw, longitud_gw, latitud, longitud)
    distances.append(dist)


def increasing_evaluation(local_dir, algorithm, model):
    header = ['ber', 'ber_th', 'snr', 'snr_db', 'snr_measured', 'snr_th_db', 'distance', 'distance_th', 'duration',
              'energy_cons', 'prr', 'pdr', 'N', 'state', 'e', 'pt']
    for n, nodes in enumerate(num_nodes):
        with open(local_dir + 'n_' + str(nodes) + '_' + algorithm + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            env.reset()
            for i, dist in enumerate(REAL_DISTANCE_NORM):
                state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[0], REAL_DISTANCE[i], 1)
                if algorithm == "SF7":
                    action = 2
                    env.step(action)
                elif algorithm == "SF12":
                    action = 0
                    env.step(action)
                else:
                    if i < 11:
                        state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[0], REAL_DISTANCE[i], nodes)
                    if 11 <= i < 22:
                        state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[1], REAL_DISTANCE[i], nodes)
                    if 22 <= i < 32:
                        state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[2], REAL_DISTANCE[i], nodes)
                    if 32 <= i < 43:
                        state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[3], REAL_DISTANCE[i], nodes)
                    if 43 <= i < 54:
                        state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[4], REAL_DISTANCE[i], nodes)
                    if 54 <= i < 65:
                        state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[5], REAL_DISTANCE[i], nodes)
                    if 65 <= i < 76:
                        state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[6], REAL_DISTANCE[i], nodes)
                    if 76 <= i <= 86:
                        state = env.set_ber_distance_snr(BER_TH[0], SNR_TH[7], REAL_DISTANCE[i], nodes)

                    action, _state = model_rec.predict(state)  # predecimos la acción más recomendada para ese estado
                    env.step(action)

                ber, ber_th, snr, snr_th, distance, distance_th, duration, c_total, prr, pdr, N, state, e, pt = env.getStatistics()
                snr_db = 10 * math.log10(snr)
                #ber_th = BER_TH[int(np.where(BER_TH_NORM == ber_norm)[0])]
                #distance_th = REAL_DISTANCE[int(np.where(REAL_DISTANCE_NORM == distance_norm)[0])]
                #snr_measured = SNR_TH[int(np.where(SNR_TH_NORM == snr_measured)[0])]
                snr_th_db = 10 * math.log10(snr_th)

                data_row = [ber, ber_th, snr, snr_db, snr_th, snr_th_db, distance, distance_th, duration, c_total, prr,
                            pdr, N, state, pt]
                writer.writerow(data_row)


model_ppo = PPO.load("logs/best_model_ppo_2.zip")
model_a2c = A2C.load("logs/best_model_rec_2.zip")
model_rec = RecurrentPPO.load("logs/best_model.zip")

BER_TH = [0.4, 0.00013895754823009532, 6.390550739301948e-05, 6.743237594015682e-05, 7.522516546093483e-06,
          1.8241669079988032e-06, 3.351781950877708e-07]

SNR_TH = [3.981071706, 1.258925412, 0.5011872336, 0.1995262315, 0.05011872336, 0.0316227766, 0.01995262315,
          0.07943282347]

BER_TH_NORM = normalize_data(BER_TH)

SNR_TH_NORM = normalize_data(SNR_TH)

REAL_DISTANCE = []

for item in distances:
    if item not in REAL_DISTANCE:
        REAL_DISTANCE.append(item)

REAL_DISTANCE_NORM = normalize_data(REAL_DISTANCE)

env = loraEnv(10)
state = env.reset()

# EVALUATE
##############################

local_dir = 'results/scenario2/'


def evaluate():
    #increasing_evaluation(local_dir, "A2C", model_a2c)
    #increasing_evaluation(local_dir, "A2C", model_a2c)
    increasing_evaluation(local_dir, "RecurrentPPO", model_rec)
    #increasing_evaluation(local_dir, "SF7", model_rec)
    #increasing_evaluation(local_dir, "SF12", model_rec)

evaluate()

n = 1
data_ppo = pd.read_csv(local_dir + 'n_'+str(n)+'_PPO.csv')
data_a2c = pd.read_csv(local_dir + 'n_'+str(n)+'_A2C.csv')
data_rec = pd.read_csv(local_dir + 'n_'+str(n)+'_RecurrentPPO.csv')
data_SF7 = pd.read_csv(local_dir + 'n_'+str(n)+'_SF7.csv')
data_SF12 = pd.read_csv(local_dir + 'n_'+str(n)+'_SF12.csv')

data = [data_rec]
#data = [data_ppo, data_a2c, data_rec, data_SF7, data_SF12]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink']
labels = ["RecPPO", "PPO", "RecPPO", "Max", "Min"]


def plot_snr_comparison():

    fig = plt.figure(figsize=(5, 4))
    ax0 = fig.subplots(1, 1)
    for i, n in enumerate(data):
        snr = smooth(n['snr_db'], 10)
        if i >= 3:
            sns.lineplot(ax=ax0, x=n.index, y=snr, data=n, label=labels[i], alpha=.5, color=colors[i], ls=':')
        else:
            sns.lineplot(ax=ax0, x=n.index, y=snr, data=n, label=labels[i], alpha=.5, color=colors[i])
    sns.lineplot(ax=ax0, x=n.index, y='snr_th_db', data=data[0], label=r'$SNR_{measured}$', alpha=1, color='red')

    ax0.text(x=4.7, y=6.3, s="A", color='red')
    ax0.text(x=15, y=1.3, s="B", color='red')
    ax0.text(x=25, y=-2.7, s="C", color='red')
    ax0.text(x=36, y=-6.7, s="D", color='red')
    ax0.text(x=47, y=-12.7, s="E", color='red')
    ax0.text(x=58, y=-14.7, s="F", color='red')
    ax0.text(x=69, y=-16.7, s="G", color='red')
    ax0.text(x=80, y=-10.7, s="H", color='red')
    ax0.legend(fontsize=9)
    ax0.set(xlabel='Uplink messages', ylabel='SNR (dB)')
    fig.tight_layout()
 #   plt.savefig('results/s2_snr.png', dpi=400)
    plt.show()


def plot_distance_comparison():
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.subplots(1, 1)
    for i, n in enumerate(data):
        distance = smooth(n['distance'], 10)
        if i >= 3:
            sns.lineplot(ax=ax1, x=n.index, y=distance, data=n, label=labels[i], alpha=.5, color=colors[i], ls=':')
        else:
            sns.lineplot(ax=ax1, x=n.index, y=distance, data=n, label=labels[i], alpha=.5, color=colors[i])
    sns.lineplot(ax=ax1, x=n.index, y='distance_th', data=data[0], label=r'$Distance_{real}$', alpha=1, color='red')
    ax1.set(xlabel='Uplink messages', ylabel='Distance (km)')
    ax1.legend(loc='center right', fontsize=9)
    fig.tight_layout()
    plt.savefig('s2_distance.png', dpi=400)
    plt.show()


def plot_cdf(data_ppo, data_a2c, data_rec):

    # sort the data:
    data_sorted_ppo = np.sort(data_ppo)
    data_sorted_a2c = np.sort(data_a2c)
    data_sorted_rec = np.sort(data_rec)

    # calculate the proportional values of samples
    p_ppo = 1. * np.arange(len(data_ppo)) / (len(data_ppo) - 1)
    p_a2c = 1. * np.arange(len(data_a2c)) / (len(data_a2c) - 1)
    p_rec = 1. * np.arange(len(data_rec)) / (len(data_rec) - 1)

    # plot the sorted data:
    fig = plt.figure(figsize=(5, 4))
    plt.plot(data_sorted_ppo, p_ppo, color='tab:blue', label='PPO')
    plt.plot(data_sorted_a2c, p_a2c, color='tab:green', label='A2C')
    plt.plot(data_sorted_rec, p_rec, color='tab:orange', label='RecurrentPPO')
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.xlabel('Estimated BER')
    plt.ylabel('$p$')
    plt.savefig('s2_cdf.png', dpi=400)
    plt.show()


def plot_energy():
    fig = plt.figure(figsize=(5, 4))
    ax0 = fig.subplots(1, 1)

    nodes = [1, 5, 10]
    alpha = [0.6, 0.6, 0.6, 0.6, 0.6]
    shift = [-1, -0.5, 0, 0.5, 1]

    for i, node in enumerate(nodes):
        data_ppo = pd.read_csv(local_dir + 'n_' + str(node) + '_PPO.csv')
        data_a2c = pd.read_csv(local_dir + 'n_' + str(node) + '_A2C.csv')
        data_rec = pd.read_csv(local_dir + 'n_' + str(node) + '_RecurrentPPO.csv')
        data_SF7 = pd.read_csv(local_dir + 'n_' + str(node) + '_SF7.csv')
        data_SF12 = pd.read_csv(local_dir + 'n_' + str(node) + '_SF12.csv')

        data = [data_a2c, data_ppo, data_rec, data_SF7, data_SF12]
        for j, n in enumerate(data):
            energy = n['energy_cons'].sum()
            if j >= 3:
                ax0.bar(node + shift[j], energy, capsize=2, width=0.5, zorder=5, edgecolor=colors[j],
                        label=labels[j] if i == 0 else "", alpha=alpha[j], hatch='//', fill=False)
            else:
                ax0.bar(node + shift[j], energy, capsize=2, width=0.5, zorder=5, color=colors[j],
                        label=labels[j] if i == 0 else "", alpha=alpha[j])
            ax0.plot([], [], lw=5, color=colors[j])

    ax0.legend(fontsize=9)
    ax0.set_xticks(nodes)
    ax0.set_ylabel(r'$Energy_{consumption} (J)$')
    ax0.set_xlabel('Nodes')
    ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig('s2_energy', dpi=400)
    plt.show()


def plot_eclipses():
    image = cv2.imread('gps.png')
    overlay = image.copy()

    # Window name in which image is displayed
    window_name = 'Image'

    center_coordinates = (375, 255)
    center_coordinates_2 = (375, 298)
    center_coordinates_3 = (375, 341)
    center_coordinates_4 = (375, 384)
    center_coordinates_5 = (375, 427)
    center_coordinates_6 = (375, 470)
    center_coordinates_7 = (375, 513)
    center_coordinates_8 = (375, 556)

    axesLength = (42, 30)
    axesLength_2 = (85, 60)
    axesLength_3 = (120, 90)
    axesLength_4 = (165, 120)
    axesLength_5 = (208, 130)
    axesLength_6 = (230, 150)
    axesLength_7 = (280, 170)
    axesLength_8 = (323, 190)

    angle = 90
    startAngle = 90
    endAngle = 450

    color = (127, 255, 0)
    color_2 = (0, 255, 0)
    color_3 = (50, 205, 50)
    color_4 = (0, 128, 0)
    color_5 = (37, 73, 141)
    color_6 = (0, 0, 255)
    color_7 = (0, 0, 200)
    color_8 = (37, 73, 141)

    # Line thickness of 5 px
    thickness = -1
    alpha = 0.75
    # Using cv2.ellipse() method
    # Draw a ellipse with red line borders of thickness of 5 px
    cv2.ellipse(image, center_coordinates_8, axesLength_8,
                angle, startAngle, endAngle, color_8, thickness)
    cv2.ellipse(image, center_coordinates_7, axesLength_7,
                angle, startAngle, endAngle, color_7, thickness)
    cv2.ellipse(image, center_coordinates_6, axesLength_6,
                angle, startAngle, endAngle, color_6, thickness)
    cv2.ellipse(image, center_coordinates_5, axesLength_5,
                angle, startAngle, endAngle, color_5, thickness)
    cv2.ellipse(image, center_coordinates_4, axesLength_4,
                angle, startAngle, endAngle, color_4, thickness)
    cv2.ellipse(image, center_coordinates_3, axesLength_3,
                angle, startAngle, endAngle, color_3, thickness)
    cv2.ellipse(image, center_coordinates_2, axesLength_2,
                angle, startAngle, endAngle, color_2, thickness)
    cv2.ellipse(image, center_coordinates, axesLength,
                angle, startAngle, endAngle, color, thickness)

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Displaying the image
    cv2.imshow(window_name, image_new)
    cv2.waitKey()
    cv2.imwrite('map_modified_eclipses.png', image_new)

def plot_points():
    image = cv2.imread('map_modified_eclipses.png')
    overlay = image.copy()
    window_name = 'Image'

    start_point = (367, 210)
    end_point = (376, 219)
    color = (0, 0, 0)
    thickness = -1

    img = cv2.rectangle(image, start_point, end_point, color, thickness)

    # img = cv2.circle(image, (372, 215), 5, (0, 0, 0), -1)

    img = cv2.circle(image, (372, 299), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (372, 380), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (386, 460), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (395, 545), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (395, 545), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (395, 631), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (377, 700), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (432, 776), 4, (0, 0, 0), -1)
    img = cv2.circle(image, (443, 858), 4, (0, 0, 0), -1)

    alpha = 0.5

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    cv2.imshow(window_name, image_new)
    cv2.waitKey()
    cv2.imwrite('map_modified_eclipses_points.png', image_new)


plot_snr_comparison()
#plot_distance_comparison()
#plot_cdf(data_ppo['ber'], data_a2c['ber'], data_rec['ber'])
#plot_energy()
#plot_eclipses()
#plot_points()