#%%-------------------------------------------IMPORT------------------------------OK
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
nest.ResetKernel()
nest.SetKernelStatus({"overwrite_files": True,"resolution": RESOLUTION})
random.seed()
seed_g = random.randint(10, 10000)
seed_r = random.randint(10, 10000)
nest.SetKernelStatus({'grng_seed' : seed_g })
nest.SetKernelStatus({'rng_seeds' : [seed_r]})
nest.set_verbosity("M_ERROR")
#**************NODS********************
sys.path.insert(1, './nods/')
from core import NODS
from utils import *
from plot import plot_cell_activity
params_filename = 'model_parameters.json'
root_path = './nods/'
with open(os.path.join(root_path,params_filename), "r") as read_file_param:
    params = json.load(read_file_param)
#**************PLOTS********************
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import seaborn as sns
with open('demo_cereb_simple.json', "r") as read_file:
    net_config = json.load(read_file)    
#%%**************NO DEPENDENCY**************
NO_dependency = False
#%%-------------------------------------------CREATE NETWORK----------------------OK
cell_types = list(net_config['cell_types'].keys())
neuronal_populations = {cell_name:{'cell_ids':[],'cell_pos':[]} for cell_name in cell_types}
for cell_name in cell_types:
    print(net_config['cell_types'][cell_name]['numerosity'],'\t',net_config['cell_types'][cell_name]['display_name'])
    nest.CopyModel( net_config['cell_types'][cell_name]['neuron_model'], cell_name)
    nest.SetDefaults(cell_name, net_config['cell_types'][cell_name]['parameters'])
    neuronal_populations[cell_name]['cell_ids'] = nest.Create(cell_name, net_config['cell_types'][cell_name]['numerosity'])
    
num_subpop = 4 #net_config['geometry']['num_subpop']
mossy_subpop = [[] for i in range(num_subpop)]
for n in range(num_subpop):
    for i in range(int(net_config['cell_types']['mossy_fibers']['numerosity']/num_subpop)*n,int(net_config['cell_types']['mossy_fibers']['numerosity']/num_subpop)*(n+1)):
        mossy_subpop[n].append(neuronal_populations['mossy_fibers']['cell_ids'][i])
#%%-------------------------------------------CONNECT NETWORK---------------------OK
connection_models = list(net_config['connection_models'].keys())
for conn_model in connection_models:
    pre_name = net_config['connection_models'][conn_model]['pre']
    post_name = net_config['connection_models'][conn_model]['post']
    print("Connecting ",pre_name," to ",post_name, "(",conn_model,")")
    plasticity = True
    if conn_model == 'parallel_fiber_to_purkinje':
        if plasticity:     
            t0 = time.time()
            vt = nest.Create("volume_transmitter_alberto", int(net_config['cell_types'][post_name]['numerosity']*(net_config['cell_types'][pre_name]['numerosity']*net_config['connection_models'][conn_model]['ratio'])))
            for n, vti in enumerate(vt):
                nest.SetStatus([vti], {"vt_num": n})
            t = time.time() - t0
            print('volume transmitter created in: ', t, ' sec')

            recdict2 = {"to_memory": False,
                        "to_file":    True,
                        "label":     "pf-PC_",
                        "senders":    neuronal_populations[pre_name]['cell_ids'],
                        "targets":    neuronal_populations[post_name]['cell_ids']}
            WeightPFPC = nest.Create('weight_recorder', params=recdict2)

            print('Set connectivity parameters')
            nest.SetDefaults(net_config['connection_models'][conn_model]['syn_spec']['model'],
                            {"A_minus": net_config['connection_models'][conn_model]['parameters']['A_minus'],
                            "A_plus":   net_config['connection_models'][conn_model]['parameters']['A_plus'],
                            "Wmin":     net_config['connection_models'][conn_model]['parameters']['Wmin'],
                            "Wmax":     net_config['connection_models'][conn_model]['parameters']['Wmax'],
                            "vt":       vt[0]#,
                            # "weight_recorder": WeightPFPC[0]
                            })  
            conn_dict={"rule": "fixed_indegree", "indegree": int(net_config['cell_types'][pre_name]['numerosity']*net_config['connection_models'][conn_model]['ratio'])}
            syn_dict=net_config['connection_models'][conn_model]['syn_spec']
            nest.Connect(neuronal_populations[pre_name]['cell_ids'], neuronal_populations[post_name]['cell_ids'], conn_spec=conn_dict, syn_spec=syn_dict)               
            vt_num = 0
            PC_vt_dict = {}
            source_ids = []
            PC_ids= []
            nos_ids = []
            t0 = time.time()        
            for i, PCi in enumerate(neuronal_populations[post_name]['cell_ids']):
                print(i) 
                C = nest.GetConnections(neuronal_populations[pre_name]['cell_ids'], [PCi])
                for n in range(len(C)):
                    nest.SetStatus([C[n]], {'vt_num': float(vt_num)})
                    if not NO_dependency:
                        nest.SetStatus([C[n]], {'meta_l': float(1.)}) 
                    source_ids.append(C[n][0])
                    PC_ids.append(C[n][1])
                    nos_ids.append(int(vt_num)) 
                    vt_num +=1
                PC_vt_dict[PCi]=np.array(nest.GetStatus(C, {'vt_num'}),dtype=int).T[0]
                #-----cf-PC connection------
                vt_tmp = [ vt[n] for n in PC_vt_dict[PCi]]
                cf_PC = nest.GetConnections(neuronal_populations['io_cell']['cell_ids'], [PCi])            
                nest.Connect([cf_PC[0][0]], vt_tmp, {'rule':'all_to_all'},{"model": "static_synapse","weight": 1.0, "delay": 1.0})
            t = (time.time() - t0)/60
            print('volume transmitter initialized in: ', t, ' min')
        else:
            conn_dict={"rule": "fixed_indegree", "indegree": int(net_config['cell_types'][pre_name]['numerosity']*net_config['connection_models'][conn_model]['ratio'])}
            syn_dict={"model": "static_synapse","weight": 0.4,"delay": 5.0}
            nest.Connect(neuronal_populations[pre_name]['cell_ids'], neuronal_populations[post_name]['cell_ids'], conn_spec=conn_dict, syn_spec=syn_dict)               

    elif conn_model == 'mossy_to_granule':
        pass 
    else:
        conn_dict=net_config['connection_models'][conn_model]['conn_spec']
        syn_dict=net_config['connection_models'][conn_model]['syn_spec']
        nest.Connect(neuronal_populations[pre_name]['cell_ids'], neuronal_populations[post_name]['cell_ids'], conn_spec=conn_dict, syn_spec=syn_dict)               


#%%-------------------------------------------DEFINE GEOMETRY---------------------OK
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import seaborn as sns
pc_color  = net_config['cell_types']['purkinje_cell']['color'][0]
grc_color  = net_config['cell_types']['granule_cell']['color'][0]
nos_color  = "#82B366"
if plasticity:
    ev_point_ids = nos_ids#[pfs[i][4] for i in range(len(pfs))]
    cluster_ev_point_ids = PC_ids#[pfs[i][1] for i in range(len(pfs))]
    cluster_nos_ids = PC_ids#[pfs[i][1] for i in range(len(pfs))]
else:
    pfs = nest.GetConnections(neuronal_populations['granule_cell']['cell_ids'], neuronal_populations['purkinje_cell']['cell_ids'])
    source_ids = [pfs[i][0] for i in range(len(pfs))]
    PC_ids = [pfs[i][1] for i in range(len(pfs))]
    nos_ids = [pfs[i][4] for i in range(len(pfs))]
    ev_point_ids = [pfs[i][4] for i in range(len(pfs))]
    cluster_ev_point_ids = [pfs[i][1] for i in range(len(pfs))]
    cluster_nos_ids = [pfs[i][1] for i in range(len(pfs))]



GR = neuronal_populations['granule_cell']['cell_ids']
PC = neuronal_populations['purkinje_cell']['cell_ids']
pfs = nest.GetConnections(GR,PC)            

# def geometry constants
y_ml = net_config['geometry']['y_ml'] # height of molecular layer
y_gl = net_config['geometry']['y_gl'] # height of granular layer
granule_density = net_config['cell_types']['granule_cell']['spatial']['density'] # volumetric GC density [GC_num/um^3]
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
#%%-------------------------------------------DEFINE SUBPOPULATION----------------OK
'''Define subpopulation'''
fig = go.Figure()
color = ['red', 'blue', 'green', 'yellow']
granule_subpop = [[] for i in range(num_subpop)]
nos_coord_subpop = [[] for i in range(num_subpop)]
x_left = np.min(nNOS_coordinates[:,0])
x_right = np.max(nNOS_coordinates[:,0])
subpop_boarder = np.linspace(x_left,x_right,num_subpop+1)
for n in range(num_subpop):
    nos_coord_subpop[n]=nNOS_coordinates[np.where((nNOS_coordinates[:,0]>subpop_boarder[n]) & (nNOS_coordinates[:,0]<subpop_boarder[n+1]))[0]]
    granule_subpop[n]=list(np.unique(np.array([source_ids[i] for i in np.where((nNOS_coordinates[:,0]>subpop_boarder[n]) & (nNOS_coordinates[:,0]<subpop_boarder[n+1]))[0]])))
    
    fig.add_trace(go.Scatter3d(x = nos_coord_subpop[n][:,0], y = nos_coord_subpop[n][:,2], z = nos_coord_subpop[n][:,1], mode = 'markers', marker = dict(size =3, color = color[n], opacity = 0.5),name = 'nos sub pop '+str(n)))
fig.add_trace(go.Scatter3d(x = PC_coord['PC_x'].to_numpy(), y =PC_coord['PC_z'].to_numpy(), z = PC_coord['PC_y'].to_numpy(), 
    mode = 'markers', marker = dict(size =20, color = pc_color, opacity = 0.8),name = 'PC'))
# 
fig.show()

conn_model='mossy_to_granule'
conn_dict=net_config['connection_models'][conn_model]['conn_spec']
syn_dict=net_config['connection_models'][conn_model]['syn_spec']             
for n in range(num_subpop):
    nest.Connect(mossy_subpop[n], granule_subpop[n], conn_spec=conn_dict, syn_spec=syn_dict)
#%%-------------------------------------------DEFINE CS STIMULI---------------------
print('CS stimulus')
CS_burst_dur = net_config['devices']['CS']['parameters']["burst_dur"]
CS_start_first = float(net_config['devices']['CS']['parameters']["start_first"])
CS_f_rate = net_config['devices']['CS']['parameters']["rate"]
CS_n_spikes = int(net_config['devices']['CS']['parameters']["rate"] * CS_burst_dur / 1000)
between_start = net_config['devices']['CS']['parameters']["between_start"]
n_trials = net_config['devices']['CS']['parameters']["n_trials"]
CS_isi = int(CS_burst_dur/CS_n_spikes)

CS_matrix_start_pre = np.round((np.linspace(100.0, 228.0, 11)))
CS_matrix_start_post = np.round((np.linspace(240.0, 368.0, 11)))
CS_matrix_first_pre = np.concatenate([CS_matrix_start_pre + between_start * t for t in range(n_trials)])
CS_matrix_first_post = np.concatenate([CS_matrix_start_post + between_start * t for t in range(n_trials)])

CS_matrix = []
for i in range(int(len(mossy_subpop[0])/2)):
    CS_matrix.append(CS_matrix_first_pre+i)
    CS_matrix.append(CS_matrix_first_post+i)

CS_device = nest.Create(net_config['devices']['CS']['device'], len(mossy_subpop[0]))

for sg in range(len(CS_device)):
    nest.SetStatus(CS_device[sg : sg + 1], params={"spike_times": CS_matrix[sg].tolist()})
nest.Connect(CS_device,mossy_subpop[0], 'all_to_all')

#%%-------------------------------------------DEFINE US STIMULUS --------------------
print('US stimulus')
US_burst_dur = net_config['devices']['US']['parameters']["burst_dur"]
US_start_first = net_config['devices']['US']['parameters']["start_first"]
US_isi = 1000 / net_config['devices']['US']['parameters']["rate"]
between_start = net_config['devices']['US']['parameters']["between_start"]
n_trials = net_config['devices']['US']['parameters']["n_trials"]
US_matrix = np.concatenate([np.arange(US_start_first, US_start_first + US_burst_dur + US_isi, US_isi)
                            + between_start * t
                            for t in range(n_trials)])
US_device = nest.Create(net_config['devices']['US']['device'], params={"spike_times": US_matrix})
nest.Connect(US_device,neuronal_populations['io_cell']['cell_ids'], conn_spec=net_config['devices']['US']['conn_spec'],syn_spec=net_config['devices']['US']['syn_spec'])

#%%-------------------------------------------DEFINE NOISE---------------------
print('background noise')
noise_device = nest.Create(net_config['devices']['background_noise']['device'], 1)
nest.Connect(noise_device,neuronal_populations['mossy_fibers']['cell_ids'], 'all_to_all')
nest.SetStatus(noise_device, params=net_config['devices']['background_noise']['parameters'])
#%%-------------------------------------------DEFINE RECORDERS---------------------
devices = list(net_config['devices'].keys())
spikedetectors = {}
for device_name in devices:
    if 'record' in device_name:
        cell_name = net_config['devices'][device_name]['cell_types']
        spikedetectors[cell_name] = nest.Create(net_config['devices'][device_name]['device'], params=net_config['devices'][device_name]['parameters'])
        nest.Connect(neuronal_populations[cell_name]['cell_ids'],spikedetectors[cell_name])


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
    sim.time = np.arange(0,1.*between_start*n_trials,1)
 
    sim.init_simulation(simulation_file,store_sim=False) # If you want to save sim inizialization change store_sim=True
    t = time.time() - t0
    print("time {}".format(t))

#%%-------------------------------------------SIMULATE NETWORK---------------------
#Simulate Network
print('simulate')
#nest.SetKernelStatus({"overwrite_files": True,"resolution": RESOLUTION,'grng_seed': 101,'rng_seeds': [100 + k for k in range(2,CORES+2)],'local_num_threads': CORES, 'total_num_virtual_procs': CORES})
print('Single trial length: ',between_start)
for trial in range(n_trials+1):
    t0 = time.time()
    print('Trial ', trial+1, 'over ', n_trials+1)
    nest.Simulate(between_start)
    t = time.time() - t0
    print('Time: ', t)

# '''Simulate network'''
# NO_threshold = 130
# processed = 0
# trial_CS_stimulus = np.copy(CS_stimulus)
# trial_US_stimulus = np.copy(US_stimulus)
# t0 = time.time()
# for trial in range(n_trial): 
#     print("Simulating trial: " + str(trial) )
#     trial_CS_stimulus[:,1] = CS_stimulus[:,1]+(trial*trial_len)   
#     trial_US_stimulus[:,1] = US_stimulus[:,1]+(trial*trial_len)
#     for k,id_input in enumerate(PG_input):
#         nest.SetStatus([PG_input[k]], params={"spike_times": trial_CS_stimulus[trial_CS_stimulus[:,0]==id_input,1], "allow_offgrid_times": True})
#     for k,id_error in enumerate(PG_error):
#         nest.SetStatus([PG_error[k]], params={"spike_times": trial_US_stimulus[trial_US_stimulus[:,0]==id_error,1], "allow_offgrid_times": True})
#     for t in range(trial*trial_len, (trial+1)*trial_len):
#         nest.Simulate(1.)
#         time.sleep(0.01)
#         if NO_dependency:    
#             ID_cell = nest.GetStatus(spikedetector_GR, "events")[0]["senders"]
#             active_sources = ID_cell[processed:]
#             processed += len(active_sources)
#             sim.evaluate_diffusion(active_sources,t)  
#             list_dict = []
#             for i in range(len(pfs)):
#                 list_dict.append({'meta_l' : float(sig(x=sim.NO_in_ev_points[t,i],A=1,B=NO_threshold))})
#             nest.SetStatus(pfs, list_dict)
# t = time.time() - t0
# print("time {}".format(t))

#%%-------------------------------------------PLOT PC SDF MEAN OVER TRIALS-------
'''Plot PC sdf mean'''
palette = list(reversed(sns.color_palette("viridis", n_trials).as_hex()))
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(vmin=0, vmax=n_trials))
cell = 'pc_spikes'
step = 50
sdf_mean_cell = []
sdf_maf_cell = []
for trial in range(n_trials):
    start = trial*between_start
    stop = CS_start_first+CS_burst_dur+trial*between_start
    spk = get_spike_activity(cell)
    sdf_cell = sdf(start=start, stop=stop, spk=spk, step=step)
    sdf_mean_cell.append(sdf_mean(sdf_cell))
    sdf_maf_cell.append(sdf_maf(sdf_cell))

fig = plt.figure()
for trial in range(n_trials):
    plt.plot(sdf_mean_cell[trial], palette[trial])
plt.title(cell)
plt.xlabel("Time [ms]")
plt.ylabel("SDF [Hz]")
plt.axvline(CS_start_first, label = "CS start", c = "grey")
plt.axvline(US_start_first-between_start, label = "US start", c = "black")
plt.axvline(CS_start_first+CS_burst_dur, label = "CS & US end ", c = "red")

#plt.xticks(np.arange(0,351,50), np.arange(50,401,50))
plt.legend()
plt.colorbar(sm, label="Trial")
plt.show()

#%%-------------------------------------------PLOT NETWORK ACTIVITY--------------TODO improve plots 
#'''
devices = list(net_config['devices'].keys())
for device_name in devices:
    if 'record' in device_name:
        cell_name = net_config['devices'][device_name]['cell_types']
        file = net_config['devices'][device_name]['parameters']['label']
        try:
            plot_cell_activity(trial_len=between_start, n_trial=n_trials, delta_t=between_start,cell_number=net_config['cell_types'][cell_name]['numerosity'], cell_name=file, freq_plot = True, png = False, scatter = False, png_scatter = False, dir='')
            plt.show()
        except:
            pass


# %%
