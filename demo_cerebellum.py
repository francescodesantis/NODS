#%%-------------------------------------------IMPORT-----------------------------
import numpy as np
import pandas as pd
import math as m
import json
import os
import sys
import random
import dill
import time

with open('./demo_cerebellum.json', "r") as json_file:
    net_config = json.load(json_file)
folder = "./demo_cerebellum_data/"
hdf5_file = "cerebellum_NODS_smaller"
network_geom_file = folder+'geom_'+hdf5_file
network_connectivity_file = folder+'conn_'+hdf5_file
neuronal_populations = dill.load(open(network_geom_file, "rb"))
connectivity = dill.load(open(network_connectivity_file, "rb"))

#**************NEST********************
import nest


nest.Install("cerebmodule")
RESOLUTION = 1.
CORES=24
VIRTUAL_CORES = 24
nest.ResetKernel()
nest.SetKernelStatus({"overwrite_files": True,"resolution": RESOLUTION})
nest.set_verbosity("M_ERROR")  # reduce plotted info
#%%**************NO DEPENDENCY**************
NO_dependency = False
#%%-------------------------------------------CREATE NETWORK---------------------
cell_types = list(net_config['cell_types'].keys())
for cell_name in cell_types:
    if cell_name == 'glomerulus':
        nest.CopyModel('parrot_neuron', cell_name)
        neuronal_populations[cell_name]['cell_ids'] = nest.Create(cell_name, net_config['cell_types'][cell_name]['numerosity'])
        print(cell_name, neuronal_populations[cell_name]['cell_ids'][0])
    else:
        nest.CopyModel('eglif_cond_alpha_multisyn', cell_name)
        nest.SetDefaults(cell_name, net_config['cell_types'][cell_name]['parameters'])
        neuronal_populations[cell_name]['cell_ids'] = nest.Create(cell_name, net_config['cell_types'][cell_name]['numerosity'])
        dVinit = [{"Vinit": np.random.uniform(net_config['cell_types'][cell_name]['parameters']['Vinit'] - 10, net_config['cell_types'][cell_name]['parameters']['Vinit'] + 10)} for _ in range(net_config['cell_types'][cell_name]['numerosity'])]
        nest.SetStatus(neuronal_populations[cell_name]['cell_ids'], dVinit)
        print(cell_name, neuronal_populations[cell_name]['cell_ids'][0])
#%%-------------------------------------------CONNECT NETWORK---------------------
connection_models = list(net_config['connection_models'].keys())
for conn_model in connection_models:
    pre = net_config['connection_models'][conn_model]['pre']
    post = net_config['connection_models'][conn_model]['post']
    print("Connecting ",pre," to ",post, "(",conn_model,")")
    if conn_model == 'parallel_fiber_to_purkinje':     
        #'''
        data_pre=connectivity[conn_model]['id_pre']
        data_post=connectivity[conn_model]['id_post']

        t0 = time.time()
        num_syn=len(data_pre)
        vt = nest.Create("volume_transmitter_alberto", int(num_syn))
        for n, vti in enumerate(vt):
            nest.SetStatus([vti], {"vt_num": n})
        t = time.time() - t0
        print('volume transmitter created in: ', t, ' sec')
        '''
        recdict2 = {"to_memory": False,
                    "to_file":    True,
                    "label":     "pf-PC_",
                    "senders":    neuronal_populations['granule_cell']['cell_ids'],
                    "targets":    neuronal_populations['purkinje_cell']['cell_ids']}
        WeightPFPC = nest.Create('weight_recorder', params=recdict2)
        # '''

        print('Set connectivity parameters')
        nest.SetDefaults(net_config['connection_models'][conn_model]['synapse_model'],
                        {"A_minus": net_config['connection_models'][conn_model]['parameters']['A_minus'],
                        "A_plus":   net_config['connection_models'][conn_model]['parameters']['A_plus'],
                        "Wmin":     net_config['connection_models'][conn_model]['parameters']['Wmin'],
                        "Wmax":     net_config['connection_models'][conn_model]['parameters']['Wmax'],
                        "vt":       vt[0]#,
                        # "weight_recorder": WeightPFPC[0]
                        })
        syn_param = {"model":  net_config['connection_models'][conn_model]['synapse_model'],
                    "weight": net_config['connection_models'][conn_model]['weight'],
                    "delay":  net_config['connection_models'][conn_model]['delay'],
                    "receptor_type": net_config['cell_types']['purkinje_cell']['receptors']['granule_cell']
                    }                    
        nest.Connect(data_pre+neuronal_populations['granule_cell']['cell_ids'][0], data_post+neuronal_populations['purkinje_cell']['cell_ids'][0], {"rule": "one_to_one"}, syn_param)       
        
        print('Get pf-PC synapses')
        t0 = time.time()
        pfs = nest.GetConnections(neuronal_populations['granule_cell']['cell_ids'], neuronal_populations['purkinje_cell']['cell_ids'])
        t = (time.time() - t0)/60
        print('connection matrix extracted in: ', t, ' min')
        PC_vt_dict = {}
        for PCi in neuronal_populations['purkinje_cell']['cell_ids']:
            PC_vt_dict[PCi] = []
            
        t0 = time.time()
        vt_num=0
        for n in range(len(pfs)):
            nest.SetStatus([pfs[n]], {'vt_num': float(vt_num)})
            vt_tmp = PC_vt_dict[pfs[n][1]]
            PC_vt_dict[pfs[n][1]] = np.append(vt_tmp,vt_num)
            vt_num += 1
            vt_num = int(vt_num)
        print('volume transmitter initialized in: ', t, ' min')

        print("Connecting io_cell to purkinje_cell (io_to_purkinje)")
        t0 = time.time()
        cf_PC = nest.GetConnections(neuronal_populations['io_cell']['cell_ids'], neuronal_populations['purkinje_cell']['cell_ids'])
        for i,syn in enumerate(cf_PC):
            vt_tmp = [ vt[n] for n in PC_vt_dict[cf_PC[i][1]]]
            nest.Connect([cf_PC[i][0]], vt_tmp, {'rule':'all_to_all'},
                                    {"model": "static_synapse",
                                    "weight": 1.0, "delay": 1.0})
        t = time.time() - t0
        print('done in: ', t, ' sec')
        #'''
    
    else:
        if post == 'glomerulus':
            syn_param = {"model": "static_synapse", 
                     "weight": net_config['connection_models'][conn_model]['weight'], 
                     "delay": net_config['connection_models'][conn_model]['delay']}
        else:
            syn_param = {"model": "static_synapse", 
                     "weight": net_config['connection_models'][conn_model]['weight'], 
                     "delay": net_config['connection_models'][conn_model]['delay'], 
                     "receptor_type": net_config['cell_types'][post]['receptors'][pre]}        
            
        if pre == 'io_cell': #guardare rule 'fixed_indegree' 1 va bene anche per bc e sc
            nest.Connect(neuronal_populations[pre]['cell_ids'], neuronal_populations[post]['cell_ids'], {'rule': 'fixed_indegree', 'indegree': 1}, syn_param)
        else:
            data_pre=connectivity[conn_model]['id_pre']
            data_post=connectivity[conn_model]['id_post']
            nest.Connect(data_pre+neuronal_populations[pre]['cell_ids'][0], data_post+neuronal_populations[post]['cell_ids'][0], {"rule": "one_to_one"}, syn_param)
'''
init_weight = [[] for i in range(len(neuronal_populations['purkinje_cell']['cell_ids']))]
for i,PC_id in enumerate(neuronal_populations['purkinje_cell']['cell_ids']):
    for j in range(len(pfs)):
        if (pfs[j][1]==PC_id):
            init_weight[i].append(nest.GetStatus([pfs[j]], {'weight'})[0][0])
init_weight = np.array(init_weight)
#'''
#%%-------------------------------------------STIMULUS GEOMETRY---------------------
# Stimulus geometry
print('stimulus geometry')
import plotly.graph_objects as go
fig = go.Figure()

radius = net_config['devices']['CS']['radius']
x = net_config['devices']['CS']['x']
z = net_config['devices']['CS']['z']
origin = np.array((x, z))

ps = neuronal_populations['glomerulus']['cell_pos']
in_range_mask = (np.sum((ps[:, [0, 2]] - origin) ** 2, axis=1) < radius ** 2)
index = np.array(neuronal_populations['glomerulus']['cell_ids'])
id_map_glom = list(index[in_range_mask])

''' Plot stimulus geometry
xpos = ps[:,0]
ypos = ps[:,2]
zpos = ps[:,1]
xpos_stim = ps[in_range_mask,0]
ypos_stim = ps[in_range_mask,2]
zpos_stim = ps[in_range_mask,1]

#fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=3,color='yellow',opacity=0.1)))
fig.add_trace(go.Scatter3d(x=xpos_stim, y=ypos_stim, z=zpos_stim, mode='markers', marker=dict(size=4,color='yellow')))

C = nest.GetConnections(neuronal_populations['glomerulus']['cell_ids'],neuronal_populations['granule_cell']['cell_ids'])
glom_ids = [C[i][0] for i in range(len(C))]
granule_ids = [C[i][1] for i in range(len(C))]
granule_ids = np.array(granule_ids)
id_map_grc = granule_ids[np.in1d(glom_ids,np.unique(id_map_glom))]
ps = neuronal_populations['granule_cell']['cell_pos']
xpos = ps[:,0]
ypos = ps[:,2]
zpos = ps[:,1]
xpos_stim = []
ypos_stim = []
zpos_stim = []
for i in id_map_grc:
    ps = neuronal_populations['granule_cell']['cell_pos'][neuronal_populations['granule_cell']['cell_ids']==i]
    xpos_stim.append(ps[0,0])
    ypos_stim.append(ps[0,2])
    zpos_stim.append(ps[0,1])
#fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=2,color='red',opacity=0.01)))
fig.add_trace(go.Scatter3d(x=xpos_stim, y=ypos_stim, z=zpos_stim, mode='markers', marker=dict(size=2,color='red')))
fig.show()

ps = neuronal_populations['purkinje_cell']['cell_pos']
xpos = ps[:,0]
ypos = ps[:,2]
zpos = ps[:,1]
fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=6,color='black')))
#'''
#%%-------------------------------------------DEFINE CS STIMULI---------------------
print('CS stimulus')
burst_dur = net_config['devices']['CS']['parameters']["burst_dur"]
start_first = float(net_config['devices']['CS']['parameters']["start_first"])
f_rate = net_config['devices']['CS']['parameters']["rate"]
n_spikes = int(net_config['devices']['CS']['parameters']["rate"] * burst_dur / 1000)
between_start = net_config['devices']['CS']['parameters']["between_start"]
n_trials = net_config['devices']['CS']['parameters']["n_trials"]
isi = int(burst_dur/n_spikes)

CS_matrix_start_pre = np.round((np.linspace(100.0, 228.0, 11)))
CS_matrix_start_post = np.round((np.linspace(240.0, 368.0, 11)))
CS_matrix_first_pre = np.concatenate([CS_matrix_start_pre + between_start * t for t in range(n_trials)])
CS_matrix_first_post = np.concatenate([CS_matrix_start_post + between_start * t for t in range(n_trials)])

CS_matrix = []
for i in range(int(len(id_map_glom)/2)):
    CS_matrix.append(CS_matrix_first_pre+i)
    CS_matrix.append(CS_matrix_first_post+i)

CS_device = nest.Create(net_config['devices']['CS']['device'], len(id_map_glom))

for sg in range(len(CS_device)):
    nest.SetStatus(CS_device[sg : sg + 1], params={"spike_times": CS_matrix[sg].tolist()})
nest.Connect(CS_device,id_map_glom, 'all_to_all')

#%%-------------------------------------------DEFINE US STIMULUS --------------------
print('US stimulus')
burst_dur = net_config['devices']['US']['parameters']["burst_dur"]
start_first = net_config['devices']['US']['parameters']["start_first"]
isi = 1000 / net_config['devices']['US']['parameters']["rate"]
between_start = net_config['devices']['US']['parameters']["between_start"]
n_trials = net_config['devices']['US']['parameters']["n_trials"]
US_matrix = np.concatenate([np.arange(start_first, start_first + burst_dur + isi, isi)
                            + between_start * t
                            for t in range(n_trials)])
US_device = nest.Create(net_config['devices']['US']['device'], params={"spike_times": US_matrix})
conn_param = {"model": "static_synapse", 
                     "weight": 1,
                     "delay": 1,
                     "receptor_type": 1}   
nest.Connect(US_device,neuronal_populations['io_cell']['cell_ids'], {'rule':'all_to_all'},conn_param)

#%%-------------------------------------------DEFINE NOISE---------------------
print('background noise')
noise_device = nest.Create(net_config['devices']['background_noise']['device'], 1)
nest.Connect(noise_device,neuronal_populations['glomerulus']['cell_ids'], 'all_to_all')
nest.SetStatus(noise_device, params={"rate": net_config['devices']['background_noise']['parameters']['rate'], "start":net_config['devices']['background_noise']['parameters']['start'], "stop":1.*between_start*n_trials})

#%%-------------------------------------------DEFINE RECORDERS---------------------
devices = list(net_config['devices'].keys())
spikedetectors = {}
for device_name in devices:
    if 'record' in device_name:
        cell_name = net_config['devices'][device_name]['cell_types']
        spikedetectors[cell_name] = nest.Create(net_config['devices'][device_name]['device'], params=net_config['devices'][device_name]['parameters'])
        nest.Connect(neuronal_populations[cell_name]['cell_ids'],spikedetectors[cell_name])

#%%-------------------------------------------INITIALIZE NODS---------------------

#%%-------------------------------------------SIMULATE NETWORK---------------------
#Simulate Network
print('simulate')
#nest.SetKernelStatus({"overwrite_files": True,"resolution": RESOLUTION,'grng_seed': 101,'rng_seeds': [100 + k for k in range(2,CORES+2)],'local_num_threads': CORES, 'total_num_virtual_procs': CORES})
print('Single trial length: ',between_start)
for trial in range(n_trials+1):
    t0 = time.time()
    print('Trial ', trial+1, 'over ', n_trials)
    nest.Simulate(between_start)
    t = time.time() - t0
    print('Time: ', t)

