# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
from matplotlib.pyplot import cm
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import random
import csv

def smooth(y, box_pts):
  box = np.ones(box_pts) / box_pts
  y_smooth = np.convolve(y, box, mode='same')
  return y_smooth

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def python_evaluation(pos, array):
    lw = [2,1,1,1,1]
    ls = ['-.', '--', ':', '-', '-', '-']
    colors = ['lightseagreen', 'grey', 'black']
    labels = ['SSFA', 'NORAC', 'FABRIC']

    ######### BR position
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid()
    for i,a in enumerate(array):
        data = []
        b = np.loadtxt(array[i]+'/br.csv')
        for j in range(0, b.shape[0]):
            data.append(b[j])
        if (i == 0):
            data = moving_average(data, 20)
            data = np.pad(data, round((len(pos) - len(data)) / 2), mode='maximum')
            data = data[:-1]
        ax1.plot(pos, data, c = colors[i], lw = lw[i], ls=ls[i])
        ax1.plot([], [], c = colors[i], label=labels[i], lw=lw[i], ls=ls[i])
        ax1.set_ylabel('Tx Rate (Hz)')
        ax1.set_xlabel('X-position on road (m)')
        ax1.set_ylim(0, 11)
        ax1.set_xlim(0, 2000)
    plt.tight_layout()
    ax1.legend(loc='best')
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/python_results1.png', dpi=400)
    plt.show()

    ######### CBR position
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for i, a in enumerate(array):
        data = []
        b = np.loadtxt(array[i] + '/cbr.csv')
        for j in range(0, b.shape[0]):
            data.append(b[j])
        ax1.plot(pos, data, c=colors[i], lw = lw[i], ls=ls[i])
        ax1.plot([], [], c=colors[i], label=labels[i], lw=lw[i], ls=ls[i])
        ax1.set_ylabel('CBR')
        ax1.set_xlabel('X-position on road (m)')
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 2000)
        #ax2.axhspan(0.6, 0.7, facecolor='g', alpha=0.05)
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.tight_layout()
    ax1.legend(loc='best')
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/python_results2.png', dpi=400)
    plt.show()

    ######### BR mid time
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for i, a in enumerate(array):
        data = []
        b = np.loadtxt(array[i] + '/br_t.csv')
        for j in range(0, b.shape[0]):
            data.append(b[j])
        ax1.plot(data, c=colors[i], lw = lw[i], ls=ls[i])
        ax1.plot([], [], c=colors[i], label=labels[i], lw=lw[i], ls=ls[i])
        ax1.set_ylabel('Tx Rate (Hz)')
        ax1.set_xlabel('# Iterations')
        ax1.set_ylim(0, 11)
        ax1.set_xlim(0, 100)
    plt.tight_layout()
    ax1.legend(loc='best')
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/python_results3.png', dpi=400)
    plt.show()

    ######### CBR mid time
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for i, a in enumerate(array):
        data = []
        b = np.loadtxt(array[i] + '/cbr_t.csv')
        for j in range(0, b.shape[0]):
            data.append(b[j])
        ax1.plot(data, c=colors[i], lw = lw[i], ls=ls[i])
        ax1.plot([], [], c=colors[i], label=labels[i], lw=lw[i], ls=ls[i])
        ax1.set_ylabel('CBR')
        ax1.set_xlabel('# Iterations')
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 100)
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.tight_layout()
    ax1.legend(loc='best')
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/python_results4.png', dpi=400)
    plt.show()

def plot_delta():
    deltas = []
    version = 4
    filename = 'ASG_SARSA_deltas_v'+str(version)+'.txt'
    with open(filename, 'r') as f:
        for line in f:
            deltas.append(float(line))
    # PLOT DELTA
    fig, ax1 = plt.subplots(figsize=(8,3))
    ax1.grid()
    ax1.plot(deltas, 'grey', alpha=0.4, lw=1)
    data = moving_average(deltas, 1000)
    data = np.array(data)-2
    ax1.plot(data, 'grey', lw=3)
    ax1.set_xlabel('# Iterations')
    ax1.set_ylabel(r'$\Delta \theta$')
    ax1.set_xlim(0, len(deltas))
    ax1.set_ylim(0,400)
    # Make the zoom-in plot:
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/delta.png', dpi=400)
    plt.show()

def plot_reward(array):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.grid()
    label = ['nn Continuous PPO', 'nn Continuous SAC']
    color = ['tab:purple', 'tab:cyan']
    for i,a in enumerate(array):
        reward = []
        version = [4,5]
        filename = 'rewards_v'+str(version[i])+'.txt'
        with open(filename, 'r') as f:
            for line in f:
                reward.append(float(line))
        ax1.plot(reward, color=color[i], alpha=0.5, lw=1)
        ax1.plot(moving_average(reward,500), color=color[i], alpha=1, lw=2, label=label[i])
    ax1.set_xlabel('# Episodes')
    ax1.set_ylabel('Average Reward')
    ax1.set_xlim(0, len(reward))
    ax1.legend()
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/reward.pdf', dpi=300)
    plt.show()

def plot_omnet1(array):
    fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    ax1.grid()
    ax2.grid()
    linew = [2,2,2,2]
    legend = ['MDPRP', 'BFPC $u_{i}$ = 10', 'BFPC $u_{i}$ = 4', 'SAE J2945/1']
    #legend = [r'\beta = 2.5', r'\beta = 2', r'\beta = 3']
    color = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:cyan']
    alpha = [0.8, 0.8, 0.8, 0.8]
    linestyle = ['-','--',':','-']
    t = 20

    for i in range(len(array)):
        ######### BR
        ##################
        y = []
        x = []
        a = np.loadtxt('omnetpp/'+array[i]+'/pos.csv', delimiter=',')
        b = np.loadtxt('omnetpp/'+array[i]+'/br.csv', delimiter=',')
        for j in range(0, b.shape[1], 2):
            y.append(b[t, j + 1])
            x.append(a[t, j + 1])
        ax1.plot(x, y, color=color[i], lw=linew[i], alpha=alpha[0], linestyle=linestyle[0])
        ax1.set_ylabel('Tx Rate (Hz)')
        ax1.set_xlabel('X-position on road (m)')
        ax1.set_ylim(0, 11)
        ax1.set_xlim(0, 2000)

        ######### CBR
        ##################
        y = []
        x = []
        a = np.loadtxt('omnetpp/' + array[i] + '/pos.csv', delimiter=',')
        b = np.loadtxt('omnetpp/'+array[i]+'/cbt.csv', delimiter=',')
        for j in range(0, b.shape[1], 2):
            x.append(a[t, j+1])
            y.append(b[t, j+1])
        ax2.plot(x, y, color = color[i], lw=linew[i], alpha = alpha[0], linestyle=linestyle[0])
        ax2.set_ylabel('CBR')
        ax2.set_xlabel('X-position on road (m)')
        ax2.set_yticks([0, 0.6, 1])
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 2000)

    plt.tight_layout()
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet1.png', dpi=400)
    plt.show()


def plot_omnet2(array):
    #fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(nrows=3, ncols=2, figsize=(8, 7))
    distances = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    label_a = ['SSFA', 'NORAC', 'FABRIC']
    lw = [2,1,1,1,1]
    ls = ['-.', '--', ':', '-', '-', '-']
    color = ['lightseagreen', 'grey', 'black']
    alpha = [0.6, 0.6, 0.6, 0.6]
    shift = [-15, -5, 5, 15]
    lw = [2,1,1,1,1,1]
    t = 29

    ######### PDR
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for a, algorithm in enumerate(array):
        data = []
        for d, distance in enumerate(distances):
            # Run
            pdr = []
            with open('omnetpp/'+ array[a] +'/pdr'+str(distance)+'.csv', 'r',
                      newline='') as csv_file:
                next(csv_file)
                reader = csv.reader(line.replace('  ', ',') for line in csv_file)
                my_list = list(reader)
                b = np.array(my_list)[:, -1]
                b = b.astype(np.float)
                y = b
            for v in range(len(b)):
                pdr.append(y[v])
            pdr_mean = np.mean(pdr)
            pdr_std = np.std(pdr)
            ax1.bar(distances[d]+shift[a], pdr_mean, yerr=pdr_std,error_kw=dict(ecolor='black',elinewidth=0.5, lolims=False), capsize=2, width = 10, zorder = 5, color=color[a], alpha=alpha[0])
        ax1.plot([], [], lw=5, color=color[a], label=label_a[a])
    ax1.set_ylabel('Packet Delivery Ratio')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 750)
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet2_pdr2.png', dpi=400)
    plt.show()

    ######### CBR
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    ax2 = ax1.twinx()
    for a, algorithm in enumerate(array):
        y = []
        x = []
        y2 = []
        c = np.loadtxt('omnetpp/'+array[a]+'/pos.csv', delimiter=',')
        b = np.loadtxt('omnetpp/'+array[a]+'/br.csv', delimiter=',')
        d = np.loadtxt('omnetpp/' + array[a] + '/cbt.csv', delimiter=',')
        for j in range(0, b.shape[1], 2):
            y.append(b[t, j + 1])
            y2.append(d[t, j + 1])
            x.append(c[t, j + 1])
        data = y
        if (a == 0):
            data = moving_average(data, 20)
            data = np.pad(data, round((len(x) - len(data)) / 2), mode='maximum')
            data = data[:-1]
        y = data
        ax1.plot(x, y2, color=color[a], lw=lw[a], ls=ls[a], label=label_a[a])
        ax2.plot(x, y, color='red', lw=lw[a], ls=ls[a])
    ax1.set_ylabel('CBR')
    ax1.set_xlabel('X-position on road (m)')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='best', ncol=3)
    ax2.set_ylim(0, 11)
    ax1.set_xlim(0, 2000)
    ax2.set_ylabel('Tx Rate (Hz)', color='red')  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor='red')
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet2_cbr.png', dpi=400)
    plt.show()

    ######### DECODED
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    shifter = [50, 100, 150]
    for a, algorithm in enumerate(array):
        # Distance
        data = []
        with open('omnetpp/'+array[a]+'/decoded.csv', 'r',
                  newline='') as csv_file:
            next(csv_file)
            reader = csv.reader(line.replace('  ', ',') for line in csv_file)
            my_list = list(reader)
            b = np.array(my_list)[:, -1]
            b = b.astype(np.float)
            y = b
        for v in range(len(b)):
            data.append(y[v])
        data_mean = np.sum(data)
        data_std = np.std(data)
        print("DECODED " + str(algorithm) + ' = ' + str(data_mean))
        ax1.bar(shifter[a], data_mean,
                error_kw=dict(ecolor='black', elinewidth=1, lolims=False), capsize=3, width=25, zorder=5,
                color=color[a], alpha=alpha[0])
        ax1.plot([], [], lw=5, color=color[a], label=label_a[a])
    ax1.set_ylabel('Decoded packets')
    #ax1.set_yticklabels(['0', '10K', '20K', '30K', '40K'])
    ax1.set_ylim(0000000, 10000000)
    ax1.set_xlim(0,200)
    ax1.set_xticklabels(('', '', 'SSFA', '', 'NORAC', '', 'FABRIC', ''))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-4, +4))
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet2_decoded.png', dpi=400)
    plt.show()

    ######### PCR
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    shifter = [50, 100, 150]
    for a, algorithm in enumerate(array):
        # Distance
        data = []
        with open('omnetpp/'+array[a]+'/pcrcacc.csv', 'r',
                  newline='') as csv_file:
            next(csv_file)
            reader = csv.reader(line.replace('  ', ',') for line in csv_file)
            my_list = list(reader)
            b = np.array(my_list)[:, -1]
            b = b.astype(np.float)
            y = b
        for v in range(len(b)):
            data.append(y[v])
        data_mean = np.mean(data)
        data_std = np.std(data)
        print("PCR " + str(algorithm) + ' = ' + str(data_mean) + ' std = ' + str(data_std))
        ax1.bar(shifter[a], data_mean, yerr=data_std,
                error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), capsize=2, width=25, zorder=5,
                color=color[a], alpha=alpha[0])
        ax1.plot([], [], lw=5, color=color[a], label=label_a[a])
    ax1.set_ylabel('Packet Collision Ratio')
    #ax1.set_yticklabels(['0', '10K', '20K', '30K', '40K'])
    ax1.set_xlim(0,200)
    ax1.set_ylim(0,0.3)
    ax1.set_xticklabels(('', '', 'SSFA', '', 'NORAC', '', 'FABRIC', ''))
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet2_pcrcacc.png', dpi=400)
    plt.show()

def plot_omnet3(array):
    #fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(nrows=3, ncols=2, figsize=(8, 7))

    distances = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    label_a = ['SSFA', 'NORAC', 'FABRIC']
    lw = [2,1,1,1,1]
    ls = ['-.', '--', ':', '-', '-', '-']
    color = ['lightseagreen', 'grey', 'black']
    alpha = [0.6, 0.6, 0.6, 0.6]
    shift = [-15, -5, 5, 15]
    lw = [2,1,1,1,1,1]

    ######### BR
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for a, algorithm in enumerate(array):
        b = np.loadtxt('omnetpp_moving/'+array[a]+'/br.csv', delimiter=',')
        y = []
        y = b[:, 621]
        x = b[:, 620]
        ax1.plot(x, y, color=color[a], lw=lw[a], ls=ls[a], label=label_a[a])
    ax1.set_ylabel('Tx Rate (Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(0, 10)
    ax1.set_xlim(0,30)
    ax1.legend(loc='best', ncol=3)
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet3_br.png', dpi=400)
    plt.show()

    ######### CBR
    ##################
    times = [2, 3, 4, 5]
    for t in times:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        ax1.grid(True)
        for a, algorithm in enumerate(array):
            x = []
            y = []
            c = np.loadtxt('omnetpp_moving/' + array[a] + '/pos.csv', delimiter=',')
            b = np.loadtxt('omnetpp_moving/' + array[a] + '/cbt.csv', delimiter=',')
            for j in range(0, c.shape[1], 2):
                if (j == 620):
                    pass
                else:
                    y.append(b[t, j + 1])
                    x.append(c[t, j + 1])
            for index, position in enumerate(x):
                if abs(position - x[index - 1]) > 20:
                    y_mean1 = y[:index]
                    y_mean2 = y[index:]
                    x1 = x[:index]
                    x2 = x[index:]
            ax1.plot(x1, y_mean1, color=color[a], lw=lw[a], linestyle=ls[a], label=label_a[a])
            ax1.plot(x2, y_mean2, color=color[a], lw=lw[a], linestyle=ls[a])
        ax1.set_ylabel('CBR')
        ax1.set_xlabel('X-position on road (m)')
        ax1.set_ylim(0, 1)
        ax1.set(frame_on=False)
        ax1.legend(loc='best')
        plt.axhspan(0.6, 1, color='red', alpha=0.05)
        plt.axhspan(0, 0.6, color='green', alpha=0.05)
        plt.title('$t_{sim}$ = '+str(t)+' s')
        plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
        plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet3_cbr_'+str(t)+'_.png', dpi=400)
        plt.show()

def plot_omnet_sumo(array):
    #fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(nrows=3, ncols=2, figsize=(8, 7))
    label_a = ['SSFA', 'NORAC', 'FABRIC']
    ls = ['-.', '--', ':', '-', '-', '-']
    color = ['lightseagreen', 'grey', 'black']
    lw = [2,1,1,1,1,1]
    distances = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    alpha = [0.6, 0.6, 0.6, 0.6]
    shift = [-15, -5, 5, 15]

    ######### PDR
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for a, algorithm in enumerate(array):
        data = []
        for d, distance in enumerate(distances):
            # Run
            pdr = []
            with open('omnetpp_sumo/' + array[a] + '/pdr' + str(distance) + '.csv', 'r',
                      newline='') as csv_file:
                next(csv_file)
                reader = csv.reader(line.replace('  ', ',') for line in csv_file)
                my_list = list(reader)
                b = np.array(my_list)[:, -1]
                b = b.astype(np.float)
                y = b
            for v in range(round(len(b)/2)):
                pdr.append(y[v])
            pdr_mean = np.mean(pdr)
            pdr_std = np.std(pdr)
            ax1.bar(distances[d] + shift[a], pdr_mean, yerr=pdr_std,
                    error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), capsize=2, width=10, zorder=5,
                    color=color[a], alpha=alpha[0])
        ax1.plot([], [], lw=5, color=color[a], label=label_a[a])
    ax1.set_ylabel('Packet Delivery Ratio')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 750)
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet_sumo_pdr.png', dpi=400)
    plt.show()

    ######### DECODED
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    shifter = [50, 100, 150]
    for a, algorithm in enumerate(array):
        # Distance
        data = []
        with open('omnetpp_sumo/' + array[a] + '/decoded.csv', 'r',
                  newline='') as csv_file:
            next(csv_file)
            reader = csv.reader(line.replace('  ', ',') for line in csv_file)
            my_list = list(reader)
            b = np.array(my_list)[:, -1]
            b = b.astype(np.float)
            y = b
        for v in range(round(len(b) / 2)):
            data.append(y[v])
        data = np.sum(data)
        print("DECODED "+str(algorithm)+' = '+ str(data))
        ax1.bar(shifter[a], data, error_kw=dict(ecolor='black', elinewidth=1, lolims=False), capsize=3, width=25, zorder=5,
                color=color[a], alpha=alpha[0])
        ax1.plot([], [], lw=5, color=color[a], label=label_a[a])
    ax1.set_ylabel('Decoded packets')
    # ax1.set_yticklabels(['0', '10K', '20K', '30K', '40K'])
    ax1.set_ylim(0, 2000000)
    ax1.set_xlim(0, 200)
    ax1.set_xticklabels(('', '', 'SSFA', '', 'NORAC', '', 'FABRIC', ''))
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-4, +4))
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet_sumo_decoded.png', dpi=400)
    plt.show()

    ######### PCR
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    shifter = [50, 100, 150]
    for a, algorithm in enumerate(array):
        # Distance
        data = []
        with open('omnetpp_sumo/' + array[a] + '/pcrcacc.csv', 'r',
                  newline='') as csv_file:
            next(csv_file)
            reader = csv.reader(line.replace('  ', ',') for line in csv_file)
            my_list = list(reader)
            b = np.array(my_list)[:, -1]
            b = b.astype(np.float)
            y = b
        for v in range(round(len(b) / 2)):
            data.append(y[v])
        data_mean = np.mean(data)
        data_std = np.std(data)
        print("PCR "+str(algorithm)+' = '+ str(data_mean)+' std = '+str(data_std))
        ax1.bar(shifter[a], data_mean, yerr=data_std,
                error_kw=dict(ecolor='black', elinewidth=0.5, lolims=False), capsize=2, width=25, zorder=5,
                color=color[a], alpha=alpha[0])
        ax1.plot([], [], lw=5, color=color[a], label=label_a[a])
    ax1.set_ylabel('Packet Collision Ratio')
    # ax1.set_yticklabels(['0', '10K', '20K', '30K', '40K'])
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 0.3)
    ax1.set_xticklabels(('', '', 'SSFA', '', 'NORAC', '', 'FABRIC', ''))
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet_sumo_pcrcacc.png', dpi=400)
    plt.show()

    ######### BR
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for a, algorithm in enumerate(array):
        b = np.loadtxt('omnetpp_sumo/'+array[a]+'/br.csv', delimiter=',', usecols=range(2))
        y = b[:, 1]
        x = b[:, 0]
        data = y
        if (a == 0):
            data = moving_average(data, 3)
            data = np.pad(data, round((len(x) - len(data)) / 2), mode='reflect')
            y = data
        ax1.plot(x, y, color=color[a], lw=lw[a], ls=ls[a], label=label_a[a])
    ax1.set_ylabel('Tx Rate (Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(0, 10)
    ax1.set_xlim(0,40)
    ax1.legend(loc='best')
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet_sumo_br.png', dpi=400)
    plt.show()

    ######### CBR
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for a, algorithm in enumerate(array):
        b = np.loadtxt('omnetpp_sumo/'+array[a]+'/cbt.csv', delimiter=',', usecols=range(2))
        y = b[:, 1]
        x = b[:, 0]
        data = y
        if (a == 0):
            data = moving_average(data, 3)
            data = np.pad(data, round((len(x) - len(data)) / 2), mode='edge')
            y = data
        ax1.plot(x, y, color=color[a], lw=lw[a], ls=ls[a], label=label_a[a])
    ax1.set_ylabel('CBR')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0,40)
    ax1.legend(loc='best')
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.axhspan(0.6, 1, color='red', alpha=0.05)
    plt.axhspan(0, 0.6, color='green', alpha=0.05)
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet_sumo_cbr.png', dpi=400)
    plt.show()

    ######### NEIGHBORS
    ##################
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    ax1.grid(True)
    for a, algorithm in enumerate(array):
        b = np.loadtxt('omnetpp_sumo/'+array[a]+'/neighbors.csv', delimiter=',', usecols=range(2))
        y = b[:, 1]
        x = b[:, 0]
        ax1.plot(x, y, color=color[a], lw=lw[a], ls=ls[a], label=label_a[a])
    ax1.set_ylabel('Neighbors')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(0, 200)
    ax1.set_xlim(0,40)
    ax1.legend(loc='best')
    plt.tight_layout(rect=[0, 0.01, 0.99, 0.99])
    plt.savefig('C:/Users/Juan/Desktop/2021-SLFA/Figures/omnet_sumo_neighbors.png', dpi=400)
    plt.show()


######################################################################
######################################################################
######################################################################

settings = np.loadtxt('params.csv')
number_of_vehicles = int(settings[0])
limit = int(settings[1])
radius = int(settings[2])
C = int(settings[3])
iters = int(settings[4])

pos = np.linspace(0, limit, number_of_vehicles)

array = ['results/approx', 'results/norac', 'results/fabric']
#python_evaluation(pos, array)

#plot_delta()

array = ['approx', 'norac', 'fabric']
plot_omnet2(array)

#plot_omnet3(array)

#plot_omnet_sumo(array)