import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import seaborn as sns
import nest


def calculate_freq(trial_len, delta_t, n_trial, ts_cell, cell_number):
    frequencies = []
    for i in range(0,trial_len*n_trial,delta_t):
        n_spikes = len([k for k in ts_cell if k<i+delta_t and k>=i])
        freq = n_spikes/(delta_t/1000*cell_number)
        frequencies.append(freq)

    return frequencies

def plot_cell_activity(trial_len, delta_t, n_trial, cell_number, 
                       cell_name, freq_plot,png, scatter, png_scatter, dir = None, trial = None):

    cell_activity = get_spike_times(cell_name, dir)
    evs_cell = cell_activity[:,0]
    ts_cell = cell_activity[:,1]

# Making plot of spiking activity of a specific cell population
    if freq_plot:    
        title_plot = cell_name + ' spiking frequency'
        title_png = title_plot + '.png'
        frequencies = calculate_freq(trial_len, delta_t, n_trial, ts_cell, cell_number)
        plt.figure(figsize=(10,8))
        plt.plot(range(0,trial_len*n_trial,delta_t), frequencies)
        plt.title(title_plot, size =25)
        plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 4),fontsize=25)
        plt.yticks(fontsize=25)

        axes = plt.gca()
        y_lims = axes.get_ylim()
        for i in range(0, n_trial*trial_len+1, trial_len):
            plt.plot([i, i], y_lims,'k',linewidth = 0.5)
        plt.xlabel('Time [ms]', size =25)
        plt.ylabel('Frequency [Hz]', size =25)
        #plt.xlim(0,2000)
        if png:
            plt.savefig(title_png)
    if png:
        plt.savefig(title_png)
# Making Scatter plot of spiking activity of specific cell type
    if scatter:
        title_plot = 'Scatter plot ' + cell_name
        title_png = title_plot + '.png'
        y_min = np.min(evs_cell)
        y = [i-y_min for i in evs_cell]
        plt.figure(figsize=(10,8))
        plt.scatter(ts_cell, y, marker='.', s = 3)
        plt.title(title_plot, size =25)
        plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 4),fontsize=25)
        plt.yticks(ticks = np.linspace(0,cell_number,10),fontsize=25)
        plt.xlabel('Time [ms]', size =25)
        plt.ylabel('Neuron ID', size =25)
        if png_scatter:
            plt.savefig(title_png)

    if trial != None:
        for i in trial:
            title_plot = 'Scatter plot ' + cell_name + ' trial '+ str(i+1)
            title_png = title_plot + '.png'
            ts_cell = ts_cell[ts_cell >= trial_len*i and ts_cell < trial_len*(i+1)]
            evs_cell = evs_cell[:len(ts_cell)]

            y_min = np.min(evs_cell)
            y = [i-y_min for i in evs_cell]
            plt.figure(figsize=(10,8))
            plt.scatter(ts_cell, y, marker='.', s = 3)
            plt.title(title_plot, size =25)
            plt.xticks(ticks=np.linspace(0, n_trial*trial_len, 4),fontsize=25)
            plt.yticks(ticks = np.linspace(0,cell_number,10),fontsize=25)
            plt.xlabel('Time [ms]', size =25)
            plt.ylabel('Neuron ID', size =25)

def heatmap_weights(connections, init_weight, plot_single_trials = False, plot_weights = False, folder = './'):

    csv_name = 'PFPC'
    print('Reading:', csv_name)
    pthDat = folder
    files = [f for f in os.listdir(pthDat) if os.path.isfile(os.path.join(pthDat,f))]
    for f in files:
        if f.startswith(csv_name):
            break

    csv_f = pthDat+f

    PFPC_weight = pd.read_csv(csv_f, sep = '\t', names = ['PF_id','PC_id', 't_ist', 'weight'],usecols=[0,1,2,3])
    connection_dict = {}
    data = {}
    for i,PC_id in enumerate(simulation.PC):
        PF_id = []
        for j in range(len(connections)):
            if (connections[j][1]==PC_id):
                PF_id.append(connections[j][0])
        PF_id = np.array(PF_id)

        connection_dict[PC_id] = {'PF_id': PF_id, 'weight_matrix': np.zeros((len(PF_id),int(simulation.n_trials*simulation.trial_len*10)))}
        connection_dict[PC_id]['weight_matrix'][:,0] = init_weight[i].T
        #df = pd.DataFrame({'PF_id': PF_id, 'initial_weights': init_weight[i].T})
        #df.to_csv('init_weights_PC'+str(PC_id))

    for connection in range(len(PFPC_weight)):
        PC_id = PFPC_weight['PC_id'][connection]
        PF_id = PFPC_weight['PF_id'][connection]
        t_ist =  int(PFPC_weight['t_ist'][connection]*10)
        weight = PFPC_weight['weight'][connection]

        ind = np.where(connection_dict[PC_id]['PF_id'] == PF_id)[0]
        connection_dict[PC_id]['weight_matrix'][ind,t_ist] = weight

    for PC_id,data in connection_dict.items():
        for i in range(data['weight_matrix'].shape[0]):
            weights = np.nonzero(data['weight_matrix'][i])[0]
            id_weight = 0
            if(weights.shape != 0):
                for id_weight in range(len(weights)):
                    if (id_weight+1)<len(weights):
                        data['weight_matrix'][i][weights[id_weight]:weights[id_weight+1]] = data['weight_matrix'][i][weights[id_weight]]
                    elif(id_weight+1==len(weights)):
                        data['weight_matrix'][i][weights[id_weight]:-1] = data['weight_matrix'][i][weights[id_weight]]

        
    for PC_id,data in connection_dict.items():
        
        weights = data['weight_matrix']
        cmap = plt.cm.get_cmap('coolwarm')

        sns.heatmap(weights, cmap = cmap, vmin = 0.0, vmax = 4.0)
        plt.xlabel(f'Time (EXP{-1} ms)')
        plt.ylabel('PF-PC connection')
        plt.title('Weight variation for synapses of PC '+str(PC_id))
        plt.savefig(str(PC_id)+'_weight_heatmap')
        plt.show()

    if plot_single_trials:   
        for trial in [(0),(1),(2)]:
            weights = data['weight_matrix'][:,trial*simulation.trial_len*10:simulation.trial_len*10 + (trial*simulation.trial_len*10)]
            cmap = plt.cm.get_cmap('coolwarm')
            for i in range(len(weights[0:50])):
                plt.plot(range(weights.shape[1]), weights[i])
            plt.show()
            sns.heatmap(weights, cmap = cmap, vmin = 0.0, vmax = 4.0)
            plt.xlabel(f'Time (EXP{-1} ms)')
            plt.ylabel('PF-PC connection')
            if i == 0:
                plt.title('Weight variation with CS-US during trial: '+str(trial+1))
                #plt.savefig('Weight_variation_CSUS_trial_'+str(trial+1)+'.png')
            else:
                plt.title('Weight variation with noise during trial: '+str(trial+1))
                #plt.savefig('Weight_variation_noise_trial_'+str(trial+1)+'.png')
            plt.show()

    if plot_weights:
        weights = connection_dict[simulation.PC[0]]['weight_matrix']

        weights_sub = weights[:len(simulation.PF_subpop[0])]
        for i in range(len(weights_sub[:20])):
            plt.plot(range(weights.shape[1]), weights_sub[i])
        plt.show()

        weights_rem = weights[len(simulation.PF_subpop[0]):]
        for i in range(len(weights_rem[:20])):
            plt.plot(range(weights.shape[1]), weights_rem[i])
        plt.show()
    
def get_spike_times(cell_name, dir = ''):
    
    cell_name = cell_name
    print('Reading:',cell_name)
    if dir == '':
        pthDat = "./"
    else:
        pthDat = dir
    files = [f for f in os.listdir(pthDat) if os.path.isfile(os.path.join(pthDat,f))]
    for f in files:
        if f.startswith(cell_name):
            break
    cell_f = open(pthDat+f,'r').read()
    cell_f = cell_f.split('\n')
    ID_cell = []
    time_cell = []
    for i in range(len(cell_f)-1):
        splitted_string = cell_f[i].split('\t')
        ID_cell.append(float(splitted_string[0]))
        time_cell.append(float(splitted_string[1]))
    neurons_activity = np.array([ID_cell,time_cell])

    return neurons_activity.T