#%%-------------------------------------------IMPORT-----------------------------
'''Import'''
import numpy as np
import pandas as pd
import math as m
import json
import os

import sys
import random
import dill
import time
import multiprocessing
#**************NEST********************
import nest
nest.Install("cerebmodule")
RESOLUTION = 1.
CORES=24
VIRTUAL_CORES = 24
nest.ResetKernel()
nest.SetKernelStatus({"overwrite_files": True,"resolution": RESOLUTION})
random.seed()
seed_g = random.randint(10, 10000)
seed_r = random.randint(10, 10000)
nest.SetKernelStatus({'grng_seed' : seed_g })
nest.SetKernelStatus({'rng_seeds' : [seed_r]})
nest.set_verbosity("M_ERROR")  # reduce plotted info

#**************NODS********************
sys.path.insert(1, './nods/')
from core import NODS
from utils import *
from plot import plot_cell_activity

params_filename = 'model_parameters.json'
root_path = './nods/'
with open(os.path.join(root_path,params_filename), "r") as read_file:
    params = json.load(read_file)

#**************PLOTS********************
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import seaborn as sns
with open('demo_single_pc_tree.json', "r") as read_file:
    net_config = json.load(read_file)
pc_color  = net_config['colors']['purkinje_cell'][0]
grc_color  = net_config['colors']['granule_cell'][0]
nos_color  = net_config['colors']['nNOS'][0]


#%%**************NO DEPENDENCY**************
NO_dependency = False
#%%-------------------------------------------CREATE NETWORK---------------------
'''Create network'''
#**************DEFINE CELL POPULATIONS********************
nest.CopyModel('iaf_cond_exp', 'granular_neuron')
nest.CopyModel('iaf_cond_exp', 'purkinje_neuron')
nest.CopyModel('iaf_cond_exp', 'olivary_neuron')
nest.CopyModel('iaf_cond_exp', 'nuclear_neuron')
nest.SetDefaults('granular_neuron', {'t_ref': 1.0,
                                        'C_m': 2.0,
                                        'V_th': -40.0,
                                        'V_reset': -70.0,
                                        'g_L': 0.2,
                                        'tau_syn_ex': 0.5,
                                        'tau_syn_in': 10.0})
nest.SetDefaults('purkinje_neuron', {'t_ref': 2.0,
                                            'C_m': 400.0,
                                            'V_th': -52.0,
                                            'V_reset': -70.0,
                                            'g_L': 16.0,
                                            'tau_syn_ex': 0.5,
                                            'tau_syn_in': 1.6})
nest.SetDefaults('olivary_neuron', {'t_ref': 1.0,
                                        'C_m': 2.0,
                                        'V_th': -40.0,
                                        'V_reset': -70.0,
                                        'g_L': 0.2,
                                        'tau_syn_ex': 0.5,
                                        'tau_syn_in': 10.0})
nest.SetDefaults('nuclear_neuron', {'t_ref': 1.0,
                                        'C_m': 2.0,
                                        'V_th': -40.0,
                                        'V_reset': -70.0,
                                        'g_L': 0.2,
                                        'tau_syn_ex': 0.5,
                                        'tau_syn_in': 10.0})
MF_num = net_config['cell_num']['MF_num']
GR_num = net_config['cell_num']['GR_num']
PC_num = net_config['cell_num']['PC_num']
IO_num = net_config['cell_num']['IO_num']
DCN_num = net_config['cell_num']['DCN_num']
MF = nest.Create("parrot_neuron", MF_num)
GR = nest.Create("granular_neuron", GR_num)
num_subpop = net_config['geometry']['num_subpop']
MF_subpop = [[] for i in range(num_subpop)]
for n in range(num_subpop):
    for i in range(int(MF_num/num_subpop)*n,int(MF_num/num_subpop)*(n+1)):
        MF_subpop[n].append(MF[i])
GR_subpop = [[] for i in range(num_subpop)]
# for n in range(num_subpop):
#     for i in range(int(GR_num/num_subpop)*n,int(GR_num/num_subpop)*(n+1)):
#         GR_subpop[n].append(GR[i])
PC = nest.Create("purkinje_neuron", PC_num)
IO = nest.Create("olivary_neuron", IO_num)
DCN = nest.Create("nuclear_neuron", DCN_num)
#**************DEFINE CONNECTIONS********************
Init_PFPC = {'distribution': 'uniform', 'low': 1.0, 'high':3.0}
LTP1 = net_config['connections']['LTP1']
LTD1 = net_config['connections']['LTD1']
Init_PCDCN = net_config['connections']['Init_PCDCN']
Init_MFDCN = net_config['connections']['Init_MFDCN']
recdict2 = {"to_memory": False,
            "to_file":    True,
            "label":     "PFPC_",
            "senders":    GR,
            "targets":    PC}

#-----pf-PC connection------
WeightPFPC = nest.Create('weight_recorder', params=recdict2)
vt = nest.Create("volume_transmitter_alberto", int(PC_num*(GR_num*0.8)))
for n, vti in enumerate(vt):
    nest.SetStatus([vti], {"vt_num": n})
nest.SetDefaults('stdp_synapse_sinexp',
                        {"A_minus":   LTD1,
                        "A_plus":    LTP1,
                        "Wmin":      0.0,
                        "Wmax":      4.0,
                        "vt":        vt[0],
                        "weight_recorder": WeightPFPC[0]
                        })
PFPC_conn_param = {"model":  'stdp_synapse_sinexp',
                    "weight": Init_PFPC,
                    "delay":  1.0}
vt_num = 0
PC_vt_dict = {}
for i, PCi in enumerate(PC):
    nest.Connect(GR, [PCi],
                        {'rule': 'fixed_indegree',
                        'indegree': int(0.8*GR_num),
                        "multapses": False},
                        PFPC_conn_param)
    C = nest.GetConnections(GR, [PCi])
    for n in range(len(C)):
        nest.SetStatus([C[n]], {'vt_num': float(vt_num)})
        if not NO_dependency:
            nest.SetStatus([C[n]], {'meta_l': float(1.)}) #********************DA COMMENTARE PER VEDERE EFFETTO META_L
        vt_num +=1
    PC_vt_dict[PCi]=np.array(nest.GetStatus(C, {'vt_num'}),dtype=int).T[0]
pfs = nest.GetConnections(GR,PC)

init_weight = [[] for i in range(len(PC))]
for i,PC_id in enumerate(PC):
    for j in range(len(pfs)):
        if (pfs[j][1]==PC_id):
            init_weight[i].append(nest.GetStatus([pfs[j]], {'weight'})[0][0])
init_weight = np.array(init_weight)

#-----PC-DCN connection------
PCDCN_conn_param = {"model": "static_synapse",
                    "weight": Init_PCDCN,
                    "delay": 1.0}
count_DCN = 0
for P in range(PC_num):
    nest.Connect([PC[P]], [DCN[count_DCN]],
                        'one_to_one', PCDCN_conn_param)
    if P % 2 == 1:
        count_DCN += 1
#-----cf-PC connection------
for i,PCi in enumerate(PC):
    vt_tmp = [ vt[n] for n in PC_vt_dict[PCi]]
    nest.Connect([IO[i]], vt_tmp, {'rule':'all_to_all'},
                            {"model": "static_synapse",
                            "weight": 1.0, "delay": 1.0})
#-----mf-DCN connection-----
MFDCN_conn_param = {"model":  "static_synapse","weight": Init_MFDCN,"delay":  10.0}
for n in range(num_subpop):    
    nest.Connect(MF_subpop[n], DCN, 'all_to_all', MFDCN_conn_param)

spikedetector_PC = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "PC"})
spikedetector_GR = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "GR"})
# spikedetector_IO = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "IO"})
# spikedetector_DCN = nest.Create("spike_detector", params={"withgid": True,"withtime": True, "to_file": True, "label": "DCN"})
nest.Connect(PC, spikedetector_PC)
nest.Connect(GR, spikedetector_GR)
# nest.Connect(IO, spikedetector_IO)
# nest.Connect(DCN, spikedetector_DCN)
#%%-------------------------------------------DEFINE PROTOCOL--------------------
'''Define protocol'''
dt = float(1)
n_trial = net_config['protocol']['n_trial']
trial_len = net_config['protocol']['trial_len']
total_sim_len = trial_len*n_trial
input_rate = net_config['protocol']['input_rate']
error_rate = net_config['protocol']['error_rate']
noise_rate = net_config['protocol']['noise_rate']
start_CS = net_config['protocol']['start_CS']
stop_CS = net_config['protocol']['stop_CS']
start_US = net_config['protocol']['start_US']
stop_US = net_config['protocol']['stop_US']
sim_time_steps = np.arange(0,total_sim_len,dt) #[ms] 
#%%-------------------------------------------DEFINE GEOMETRY--------------------
'''Define geometry'''
source_ids = [pfs[i][0] for i in range(len(pfs))]
PC_ids = [pfs[i][1] for i in range(len(pfs))]
nos_ids = [pfs[i][4] for i in range(len(pfs))]
ev_point_ids = [pfs[i][4] for i in range(len(pfs))]
cluster_ev_point_ids = [pfs[i][1] for i in range(len(pfs))]
cluster_nos_ids = [pfs[i][1] for i in range(len(pfs))]

# def geometry constants
y_ml = net_config['geometry']['y_ml'] # height of molecular layer
y_gl = net_config['geometry']['y_gl'] # height of granular layer
granule_density = net_config['geometry']['granule_density'] # volumetric GC density [GC_num/um^3]
nos_n = int(len(pfs)/len(PC)) # calculate number of PF-PC synapses for each PC
nos_density = net_config['geometry']['nos_density']
A = nos_n/nos_density # Compute area given the number of synapses (nos_n)
x_scaffold = A/y_ml # width of the scaffold (PC tree width)
z_scaffold = (len(GR)/granule_density)/(x_scaffold*y_gl) # thickness of the scaffold
dim_ml = np.array([x_scaffold,y_ml,z_scaffold]) # array of molecular layer dimensions ?SERVONO
dim_gl = np.array([x_scaffold,y_gl,z_scaffold]) # array of granular layer dimensions ?SERVONO
# placing Granule Cells
GR_coord = pd.DataFrame({'GR_id':GR, 'GR_x':np.random.uniform(0,x_scaffold,len(GR)), 
                        'GR_y':np.random.uniform(0,y_gl,len(GR)),
                        'GR_z':np.random.uniform(0,z_scaffold,len(GR))})
# placing Purkinje Cells
PC_dist = net_config['geometry']['PC_dist']
PC_coord = pd.DataFrame({'PC_id':PC, 'PC_x':np.random.normal(x_scaffold/2,x_scaffold/4,len(PC)), 
                        'PC_y':y_gl+PC_dist/2,
                        'PC_z':np.linspace(0,PC_dist*(len(PC)-1),len(PC))})
x_nos_variation = 1
z_nos_variation = 1
y_nos_variation = 1
pc_soma = 20.0
nNOS_coordinates = np.zeros((len(pfs),3))
for i in range(len(pfs)):
    GR_x = GR_coord['GR_x'][GR_coord['GR_id']==source_ids[i]].to_numpy()[0]
    GR_y = GR_coord['GR_y'][GR_coord['GR_id']==source_ids[i]].to_numpy()[0]
    proportion = (GR_y/y_ml)*y_ml
    PC_y = PC_coord['PC_y'][PC_coord['PC_id']==PC_ids[i]].to_numpy()[0]
    PC_z = PC_coord['PC_z'][PC_coord['PC_id']==PC_ids[i]].to_numpy()[0]
    nNOS_coordinates[i,0] = random.uniform(GR_x - x_nos_variation, GR_x + x_nos_variation)
    nNOS_coordinates[i,1] = random.uniform(PC_y + pc_soma + proportion - y_nos_variation, PC_y + pc_soma + proportion + y_nos_variation)
    nNOS_coordinates[i,2] = random.uniform(PC_z - z_nos_variation/2, PC_z + z_nos_variation/2)
ev_point_coordinates = nNOS_coordinates
#%%-------------------------------------------DEFINE SUBPOPULATION---------------
'''Define subpopulation'''
fig = go.Figure()
color = ['red', 'blue', 'green', 'yellow']
GR_subpop = [[] for i in range(num_subpop)]
nos_coord_subpop = [[] for i in range(num_subpop)]
x_left = np.min(nNOS_coordinates[:,0])
x_right = np.max(nNOS_coordinates[:,0])
subpop_boarder = np.linspace(x_left,x_right,num_subpop+1)
for n in range(num_subpop):
    nos_coord_subpop[n]=nNOS_coordinates[np.where((nNOS_coordinates[:,0]>subpop_boarder[n]) & (nNOS_coordinates[:,0]<subpop_boarder[n+1]))[0]]
    GR_subpop[n]=list(np.unique(np.array([source_ids[i] for i in np.where((nNOS_coordinates[:,0]>subpop_boarder[n]) & (nNOS_coordinates[:,0]<subpop_boarder[n+1]))[0]])))
    
    fig.add_trace(go.Scatter3d(x = nos_coord_subpop[n][:,0], y = nos_coord_subpop[n][:,2], z = nos_coord_subpop[n][:,1], mode = 'markers', marker = dict(size =3, color = color[n], opacity = 0.5),name = 'nos sub pop '+str(n)))
fig.add_trace(go.Scatter3d(x = PC_coord['PC_x'].to_numpy(), y =PC_coord['PC_z'].to_numpy(), z = PC_coord['PC_y'].to_numpy(), 
    mode = 'markers', marker = dict(size =20, color = pc_color, opacity = 0.8),name = 'PC'))
# 
fig.show()
MFGR_conn_param = {"model": "static_synapse",
                   "weight": {'distribution': 'uniform',
                              'low': 2.0, 'high': 3.0},
                    "delay": 1.0}
for n in range(num_subpop):
    nest.Connect(MF_subpop[n], GR_subpop[n], {'rule': 'fixed_indegree',
                                        'indegree': 4,
                                        "multapses": False}, MFGR_conn_param)
#%%-------------------------------------------DEFINE STIMULI---------------------
'''Define stimuli'''
conn_param = {"model":  "static_synapse",
                    "weight": 1.,
                    "delay":  1.}
PG_input = nest.Create("spike_generator", MF_num)
nest.Connect(PG_input, MF, 'one_to_one',conn_param)
if num_subpop !=1:
    PG_active = []
    for i in [0]: 
        for mf_id in MF_subpop[i]: 
            A = nest.GetConnections(PG_input,[mf_id])
            PG_active.append(A[0][0])
    PG_input = PG_active

PG_error = nest.Create("spike_generator", IO_num)
nest.Connect(PG_error,IO, 'one_to_one',conn_param)

PG_noise = nest.Create("poisson_generator", 1)
nest.Connect(PG_noise,MF, 'all_to_all',conn_param)
nest.SetStatus(PG_noise, params={"rate": noise_rate, "start":0., "stop":1.*trial_len*n_trial})

bin_size = 1/1000
# CS spiking pattern
CS_pattern = []
CS_id = []
for index,mf in enumerate(PG_input):    
    CS_spikes = homogeneous_poisson_for_nest(input_rate, (stop_CS-start_CS)/1000, bin_size)
    t = np.arange(len(CS_spikes)) * bin_size
    CS_time_stamps = t[CS_spikes]*1000+start_CS
    CS_id = np.append(CS_id,np.ones(len(CS_time_stamps))*mf)
    CS_pattern = np.append(CS_pattern,CS_time_stamps)
CS_stimulus = np.zeros((len(CS_pattern),2))
CS_stimulus[:,0] = CS_id
CS_stimulus[:,1] = CS_pattern
# US spiking pattern
US_pattern = []
US_id = []
US_time_stamps = np.arange(start_US,stop_CS,1000/error_rate)
for index,cf in enumerate(PG_error):    
    US_id = np.append(US_id,np.ones(len(US_time_stamps))*cf)
    US_pattern = np.append(US_pattern,US_time_stamps)
US_stimulus = np.zeros((len(US_pattern),2))
US_stimulus[:,0] = US_id
US_stimulus[:,1] = US_pattern
#%%-------------------------------------------INITIALIZE NODS--------------------
'''Initialize NODS'''
if NO_dependency: 
    t0 = time.time()
    init_new_sim = True 
    simulation_file = 'p'
    sim = NODS(params)
    if init_new_sim:
        sim.init_geometry(nNOS_coordinates=nNOS_coordinates, ev_point_coordinates=ev_point_coordinates, source_ids=source_ids, ev_point_ids=ev_point_ids, nos_ids=nos_ids, cluster_ev_point_ids=cluster_ev_point_ids, cluster_nos_ids=cluster_nos_ids)
    else:
        sim = sim.load_simulation(simulation_file=simulation_file)  
    sim.time = sim_time_steps  
    sim.init_simulation(simulation_file,store_sim=False) # If you want to save sim inizialization change store_sim=True
    t = time.time() - t0
    print("time {}".format(t))
#%%-------------------------------------------SIMULATE NETWORK-------------------
'''Simulate network'''
NO_threshold = 130
processed = 0
trial_CS_stimulus = np.copy(CS_stimulus)
trial_US_stimulus = np.copy(US_stimulus)
t0 = time.time()
for trial in range(n_trial): 
    print("Simulating trial: " + str(trial) )
    trial_CS_stimulus[:,1] = CS_stimulus[:,1]+(trial*trial_len)   
    trial_US_stimulus[:,1] = US_stimulus[:,1]+(trial*trial_len)
    for k,id_input in enumerate(PG_input):
        nest.SetStatus([PG_input[k]], params={"spike_times": trial_CS_stimulus[trial_CS_stimulus[:,0]==id_input,1], "allow_offgrid_times": True})
    for k,id_error in enumerate(PG_error):
        nest.SetStatus([PG_error[k]], params={"spike_times": trial_US_stimulus[trial_US_stimulus[:,0]==id_error,1], "allow_offgrid_times": True})
    for t in range(trial*trial_len, (trial+1)*trial_len):
        nest.Simulate(1.)
        time.sleep(0.01)
        if NO_dependency:    
            ID_cell = nest.GetStatus(spikedetector_GR, "events")[0]["senders"]
            active_sources = ID_cell[processed:]
            processed += len(active_sources)
            sim.evaluate_diffusion(active_sources,t)  
            list_dict = []
            for i in range(len(pfs)):
                list_dict.append({'meta_l' : float(sig(x=sim.NO_in_ev_points[t,i],A=1,B=NO_threshold))})
            nest.SetStatus(pfs, list_dict)
t = time.time() - t0
print("time {}".format(t))

#%%-------------------------------------------PLOT PC SDF MEAN OVER TRIALS-------
'''Plot PC sdf mean'''
palette = list(reversed(sns.color_palette("viridis", n_trial).as_hex()))
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(vmin=0, vmax=n_trial))
cell = 'PC'
step = 50
sdf_mean_cell = []
sdf_maf_cell = []
for trial in range(n_trial):
    start = trial*trial_len
    stop = stop_CS+trial*trial_len
    spk = get_spike_activity(cell)
    sdf_cell = sdf(start=start, stop=stop, spk=spk, step=step)
    sdf_mean_cell.append(sdf_mean(sdf_cell))
    sdf_maf_cell.append(sdf_maf(sdf_cell))

fig = plt.figure()
for trial in range(n_trial):
    plt.plot(sdf_mean_cell[trial], palette[trial])
plt.title(cell)
plt.xlabel("Time [ms]")
plt.ylabel("SDF [Hz]")
plt.axvline(start_CS, label = "CS start", c = "grey")
plt.axvline(start_US, label = "US start", c = "black")
plt.axvline(stop_CS, label = "CS & US end ", c = "red")

#plt.xticks(np.arange(0,351,50), np.arange(50,401,50))
plt.legend()
plt.colorbar(sm, label="Trial")
plt.show()
#%%-------------------------------------------PLOT INTERCELLS VARIABILITY--------
'''Plot PC sdf'''
fig,axes = plt.subplots(n_trial,1, figsize=(6,20), constrained_layout =True, sharex=True, sharey=True)
for trial in range(n_trial):
    start = trial*trial_len
    stop = stop_CS+trial*trial_len
    spk = get_spike_activity(cell)
    sdf_cell = sdf(start=start, stop=stop, spk=spk, step=step)
    axes[trial].plot(sdf_cell[0], palette[trial])
    axes[trial].plot(sdf_cell[1], palette[trial])
    axes[trial].axvline(start_CS, label = "CS start", c = "grey")
    axes[trial].axvline(start_US, label = "US start", c = "black")
    axes[trial].axvline(stop_CS, label = "CS & US end ", c = "red")
    axes[trial].set_xlabel('Time [ms]')
    axes[trial].set_ylabel('SDF [Hz]')
    axes[trial].set_title('Trial '+str(trial))


#%%-------------------------------------------PLOT NETWORK ACTIVITY--------------TODO improve plots 
'''
plot_cell_activity(trial_len=trial_len, n_trial=n_trial, delta_t=5,cell_number=GR_num, cell_name='GR', freq_plot = True, png = False, scatter = True, png_scatter = False, dir='')
plt.show()
#'''
plot_cell_activity(trial_len=trial_len, n_trial=n_trial, delta_t=trial_len,cell_number=PC_num, cell_name='PC', freq_plot = True, png = False, scatter = True, png_scatter = False, dir = '')
plt.show()
'''
plot_cell_activity(trial_len=trial_len, n_trial=n_trial, delta_t=5,cell_number=IO_num, cell_name='IO', freq_plot = True, png = False, scatter = True, png_scatter = False, dir = '')
plt.show()
#'''
#%%-------------------------------------------PLOT [NO]--------------------------TODO plot NO diffusion on the dendritic tree (geometrical representation)
# '''
fig2,axes2 = plt.subplots(1,1, figsize=(15,5), constrained_layout =True, sharey=True)
probe_id = 240
probe = sim.NO_in_ev_points[:,probe_id]
axes2.plot(sim_time_steps, probe, linewidth=3, label="probe #{}".format(probe_id))
axes2.grid()
axes2.set_xlabel('time [ms]')
axes2.set_ylabel('concentration [pM]')
axes2.legend()
fig2.show()
#'''
#%%-------------------------------------------PLOT WEIGHT CHANGE AND [NO]-------- TODO da sistemare
#'''
import seaborn as sns
csv_name = 'PFPC'
print('Reading:', csv_name)
pthDat = "./"
files = [f for f in os.listdir(pthDat) if os.path.isfile(os.path.join(pthDat,f))]
for f in files:
    if f.startswith(csv_name):
        break

csv_f = pthDat+f
# extract dataframe of PFPC weight update
PFPC_weight = pd.read_csv(csv_f, sep = '\t', names = ['PF_id','PC_id', 't_ist', 'weight'],usecols=[0,1,2,3])
connection_dict = {}
data = {}
# assign to each PC its pfs

for i,PC_id in enumerate(PC):
    PF_id = []
    for j in range(len(pfs)):
        if (pfs[j][1]==PC_id):
            PF_id.append(pfs[j][0])
    PF_id = np.array(PF_id)

# initialize the connection dictionary, composed for each PC by:
# - array of connected pf
# - matrix of weights for connections (rows) in time (columns) with rows ordered as the PF_id array
    
    connection_dict[PC_id] = {'PF_id': PF_id, 'weight_matrix': np.zeros((len(PF_id),int(n_trial*trial_len*10)))}
    
# assign to the first column at t = 0 the initial weights
    connection_dict[PC_id]['weight_matrix'][:,0] = init_weight[i].T

# extraction and assignment of values of weight changes at prescise time istant (t ist)
for connection in range(len(PFPC_weight)):
    PC_id = PFPC_weight['PC_id'][connection]
    PF_id = PFPC_weight['PF_id'][connection]
    t_ist =  int(PFPC_weight['t_ist'][connection]*10)
    weight = PFPC_weight['weight'][connection]
    
    # search the index of each pf and assign the weight value update at time t_ist 
    ind = np.where(connection_dict[PC_id]['PF_id'] == PF_id)[0]
    connection_dict[PC_id]['weight_matrix'][ind,t_ist] = weight

# filling the matrix expanding the setted weight values until the weight updates

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

for PC_id,data_pfpc in connection_dict.items():
    fig3,axes3 = plt.subplots(num_subpop,2,figsize=(12,10), constrained_layout =True, sharey=True)
    weights = data['weight_matrix']
    cmap = plt.cm.get_cmap('coolwarm')
    for i in range(num_subpop):
        weights_subpop = np.isin(data_pfpc['PF_id'], np.array(GR_subpop[i]))
        weights_hm = weights[weights_subpop]

        sns.heatmap(ax = axes3[i,0], data = weights_hm, cmap = cmap, vmin = 0.0, vmax = 4.0)
        #axes3.set_xlabel(f'Time (EXP{-1} ms)')
        #axes3.set_ylabel('PF-PC connection')
        #axes3.title('Weight variation for synapses of PC ' + str(PC_id)': subpop ' + str(i+1))
        """NO_in_ev_points_T = sim.NO_in_ev_points.T
        sns.heatmap(ax = axes3[i,1], data = NO_in_ev_points_T[weights_subpop)])"""
    fig3.show()

#fig4,axes4 = plt.subplots(num_subpop,2,figsize=(15,5), constrained_layout =True, sharey=True)
#'''