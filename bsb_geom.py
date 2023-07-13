#%%-------------------------------------------IMPORT-----------------------------
'''Import'''
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
#**************BSB********************
from bsb.core import from_storage
network = from_storage("cerebellum_NODS.hdf5")

with open('demo_cerebellum.json', "r") as json_file:
    net_config = json.load(json_file)
#**************NEST********************
import nest
nest.Install("cerebmodule")
RESOLUTION = 1.
nest.SetKernelStatus({"overwrite_files": True,"resolution": RESOLUTION})
random.seed()
seed_g = random.randint(10, 10000)
seed_r = random.randint(10, 10000)
nest.SetKernelStatus({'grng_seed' : seed_g })
nest.SetKernelStatus({'rng_seeds' : [seed_r]})
nest.set_verbosity("M_ERROR")  # reduce plotted info
#%%**************NO DEPENDENCY**************
NO_dependency = True
#%%-------------------------------------------CREATE NETWORK---------------------
# definire modelli dei neuroni
cell_types = list(net_config['cell_types'].keys())
neuronal_populations = {cell_name:{'cell_ids':[],'cell_pos':[]} for cell_name in cell_types}
for cell_name in cell_types:
    if cell_name != 'glomerulus' and cell_name != 'mossy_fibers':
        nest.CopyModel('eglif_cond_alpha_multisyn', cell_name)
        nest.SetDefaults(cell_name, net_config['cell_types'][cell_name]['parameters'])
        neuronal_populations[cell_name]['cell_ids'] = nest.Create(cell_name, net_config['cell_types'][cell_name]['numerosity'])
        dVinit = [{"Vinit": np.random.uniform(net_config['cell_types'][cell_name]['parameters']['Vinit'] - 10, net_config['cell_types'][cell_name]['parameters']['Vinit'] + 10)} for _ in range(net_config['cell_types'][cell_name]['numerosity'])]
        nest.SetStatus(neuronal_populations[cell_name]['cell_ids'], dVinit)
    else:
        nest.CopyModel('parrot_neuron', cell_name)
        neuronal_populations[cell_name]['cell_ids'] = nest.Create(cell_name, net_config['cell_types'][cell_name]['numerosity'])

    # assegnare gli ids generati da nest alle coordinate spaziali generati da BSB 
    try:
        neuronal_populations[cell_name]['cell_pos'] = network.get_placement_set(cell_name).load_positions()
    except Exception as e:
        print(f"{str(e)} population has not been positioned")
#%%-------------------------------------------CONNECT NETWORK---------------------
# connettere ciclando su connessioni fatte da BSB
# ciclo su connection matrix da BSB, id di BSB sono inidici che posso usare nella lista di indici nest quindi connetto pop_post[index_BSB_pre] con pop_pre[index_BSB_post]
connection_models = list(net_config['connection_models'].keys())
for conn_model in connection_models:
    pre = net_config['connection_models'][conn_model]['pre']
    post = net_config['connection_models'][conn_model]['post']
    if conn_model != 'parallel_fiber_to_purkinje': #manca da connettere pf e pc plasticità
        print("connecting ",pre," to ",post)
        if post == 'glomerulus':
            syn_param = {"model": "static_synapse", 
                     "weight": net_config['connection_models'][conn_model]['weight'], 
                     "delay": net_config['connection_models'][conn_model]['delay']}
        else:
            syn_param = {"model": "static_synapse", 
                     "weight": net_config['connection_models'][conn_model]['weight'], 
                     "delay": net_config['connection_models'][conn_model]['delay'], 
                     "receptor_type": net_config['cell_types'][post]['receptors'][pre]}
        try:
            cs= network.get_connectivity_set(conn_model).load_connections().as_globals()
            print("using BSB connections")
            data=cs.all()
            data_pre=data[0]
            data_post=data[1]
            for i in range(len(data_pre)):
                nest.Connect([neuronal_populations[pre]['cell_ids'][data_pre[i,0]]], [neuronal_populations[post]['cell_ids'][data_post[i,0]]], {"rule": "one_to_one"}, syn_param)
        except Exception as e:
            print("no BSB connectivity")
            nest.Connect(neuronal_populations[pre]['cell_ids'], neuronal_populations[post]['cell_ids'], {"rule": "all_to_all"}, syn_param)
        

# verificare che con get.Connection si ritrovino cose sensate
# anche quando connetto PC, lo faccio per singola syn e assegno già coordinata nNOS e ev_point
#%%-------------------------------------------STIMULUS GEOMETRY---------------------
# geometria stimolo
import plotly.graph_objects as go
fig = go.Figure()

radius = net_config['devices']['CS']['radius']
x = network.configuration.network.x-(network.configuration.network.x/2)
z = network.configuration.network.z-(network.configuration.network.z/2)
origin = np.array((x, z))

ps = network.get_placement_set("glomerulus").load_positions()
ps = np.array(ps)
in_range_mask = (np.sum((ps[:, [0, 2]] - origin) ** 2, axis=1) < radius ** 2)
index = np.array(neuronal_populations['glomerulus']['cell_ids'])
id_map_glom = index[in_range_mask]
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

xpos_stim = []
ypos_stim = []
zpos_stim = []
for i in id_map_grc:
    ps = neuronal_populations['granule_cell']['cell_pos'][neuronal_populations['granule_cell']['cell_ids']==i]
    xpos_stim.append(ps[0,0])
    ypos_stim.append(ps[0,2])
    zpos_stim.append(ps[0,1])
fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=2,color='red',opacity=0.01)))
fig.add_trace(go.Scatter3d(x=xpos_stim, y=ypos_stim, z=zpos_stim, mode='markers', marker=dict(size=2,color='red')))

# %%
