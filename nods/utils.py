import numpy as np
import os 
import pandas as pd

# nNOS stimulus poisson spike-train pattern
def homogeneous_poisson(rate, start, stop, bin_size, dim):
    tmax = stop-start
    spikes = np.zeros(dim)
    nbins = np.floor(tmax/bin_size).astype(int)
    prob_of_spike = rate * bin_size
    s = np.random.rand(nbins) < prob_of_spike
    spikes[start:stop]=s
    return spikes

def homogeneous_poisson_for_nest(rate, tmax, bin_size):
    nbins = np.floor(tmax/bin_size).astype(int)
    prob_of_spike = rate * bin_size
    spikes = np.random.rand(nbins) < prob_of_spike
    return spikes

def CS_pattern(PG_input, input_rate, start_CS, stop_CS, bin_size):
    # CS spiking pattern
    CS_pattern = []
    CS_id = []
    for index,mf in enumerate(PG_input):    
        nbins = np.floor(((stop_CS-start_CS)/1000)/bin_size).astype(int)
        prob_of_spike = input_rate * bin_size
        CS_spikes = np.random.rand(nbins) < prob_of_spike
        t = np.arange(len(CS_spikes)) * bin_size
        CS_time_stamps = t[CS_spikes]*1000+start_CS
        CS_id = np.append(CS_id,np.ones(len(CS_time_stamps))*mf)
        CS_pattern = np.append(CS_pattern,CS_time_stamps)

    CS_stimulus = np.zeros((len(CS_pattern),2))
    CS_stimulus[:,0] = CS_id
    CS_stimulus[:,1] = CS_pattern

    return CS_stimulus

def US_pattern(PG_error, error_rate, start_US, stop_US):
    # US spiking pattern
    US_pattern = []
    US_id = []
    US_time_stamps = np.arange(start_US,stop_US,1000/error_rate)

    for index,cf in enumerate(PG_error):    
        US_id = np.append(US_id,np.ones(len(US_time_stamps))*cf)
        US_pattern = np.append(US_pattern,US_time_stamps)

    US_stimulus = np.zeros((len(US_pattern),2))
    US_stimulus[:,0] = US_id
    US_stimulus[:,1] = US_pattern

    return US_stimulus

def sig(x, A=2, B=170, C=5):
    return A/(1+np.exp(-(x-B)/C)) 
# nNOS stimulus regular spike-train pattern
def regular_spikes(start,stop,dt,input_rate,dim):
    #nbins = np.floor((stop-start)/dt).astype(int)
    spikes = np.zeros(dim)
    time_stamps = np.arange(start/dt,stop/dt,1000/input_rate/dt, dtype=int)
    spikes[time_stamps]=1
    return spikes

def get_source_activity(cell_name, dir = ''):

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
        
    sources_activity = pd.DataFrame({'source_id':ID_cell,'spike_time':time_cell})

    return sources_activity

def get_spike_values(nest, sd_list, pop_names):
    """ Function to select spike idxs and times from spike_det events
        Returns a list of dictionaries with spikes and times  """
    dic_list = []
    for sd, name in zip(sd_list, pop_names):
        spikes = nest.GetStatus(sd, "events")[0]["senders"]
        times = nest.GetStatus(sd, "events")[0]["times"]
        dic = {'times': times, 'neurons_idx': spikes, 'compartment_name': name}
        dic_list = dic_list + [dic]
    return dic_list

def get_weights_values(nest, weights_recorder):
    """ Function to select mean voltage and time from voltmeter events
        Returns a list of dictionaries with potentials and times  """

    dic_list = []

    weights = nest.GetStatus(weights_recorder, "events")[0]["weights"]
    times = nest.GetStatus(weights_recorder, "events")[0]["times"]
    senders = nest.GetStatus(weights_recorder, "events")[0]["senders"]
    targets = nest.GetStatus(weights_recorder, "events")[0]["targets"]

    # for s_i, t_i in zip([67838, 22216, 80039], [95457, 95457, 95525]):
    for s_i, t_i in zip([7714, 19132], [95514, 95473]):
        idx = [s == s_i and t == t_i for s, t in zip(senders, targets)]
        dic = {'times': times[idx], 'weights': weights[idx], 'sender_receiver': f's = {s_i}, t = {t_i}'}
        dic_list = dic_list + [dic]

    return dic_list