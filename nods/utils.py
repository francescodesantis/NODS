import numpy as np
import os 
import pandas as pd
import math

#TODO 
def sdf(start, stop, spk = None, g_size=20, n_first=None, step=50 ):
    '''Compute sdf for each neurons inside an interval'''  
    if n_first is not None:
        neurons = np.unique(spk[:,0])[:200]
    else:
        neurons = np.unique(spk[:,0])
    spk_first = spk[(spk[:,1]>=start-step) & (spk[:,1]<stop+step)]
    spk_first[:,1] -= start-step
    dur = stop-start+2*step
    sdf_full = np.empty([len(neurons),int(dur)])
    sdf = []
    for neu in range(len(neurons)):
        spike_times_first = spk_first[spk_first[:,0]==neurons[neu],1]
        for t in range(int(dur)):
            tau_first = t-spike_times_first
            sdf_full[neu,t] = sum(1/(math.sqrt(2*math.pi)*g_size)*np.exp(-np.power(tau_first,2)/(2*(g_size**2))))*(10**3)
        sdf.append(sdf_full[neu])
    return(sdf)
def sdf_mean(sdf):
    sdf_mean = np.mean(sdf, axis=0)

    return(sdf_mean)
def sdf_maf(sdf,  step=100):
    sdf_maf = np.convolve(sdf_mean(sdf), np.ones(step), 'valid') / step
    return(sdf_maf)

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
def regular_spikes(start,stop,dt,input_rate,dim):
    #nbins = np.floor((stop-start)/dt).astype(int)
    spikes = np.zeros(dim)
    time_stamps = np.arange(start/dt,stop/dt,1000/input_rate/dt, dtype=int)
    spikes[time_stamps]=1
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

def get_spike_activity(cell_name, path = ''):

    #print('Reading:',cell_name)
    if path == '':
        pthDat = "./"
    else:
        pthDat = path
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
        
    sources_activity_df = pd.DataFrame({'source_id':ID_cell,'spike_time':time_cell})
    neurons_activity = np.array([ID_cell,time_cell])
    return neurons_activity.T
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