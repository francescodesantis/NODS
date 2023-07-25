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
network = from_storage("cerebellum_NODS_smaller.hdf5")

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
#%%-------------------------------------------CREATE NETWORK---------------------
# definire modelli dei neuroni
cell_types = list(net_config['cell_types'].keys())
neuronal_populations = {cell_name:{'cell_ids':[],'cell_pos':[]} for cell_name in cell_types}
for cell_name in cell_types:
    if cell_name == 'glomerulus':
        nest.CopyModel('parrot_neuron', cell_name)
        neuronal_populations[cell_name]['cell_ids'] = nest.Create(cell_name, net_config['cell_types'][cell_name]['numerosity'])
    else:
        nest.CopyModel('eglif_cond_alpha_multisyn', cell_name)
        nest.SetDefaults(cell_name, net_config['cell_types'][cell_name]['parameters'])
        neuronal_populations[cell_name]['cell_ids'] = nest.Create(cell_name, net_config['cell_types'][cell_name]['numerosity'])
        dVinit = [{"Vinit": np.random.uniform(net_config['cell_types'][cell_name]['parameters']['Vinit'] - 10, net_config['cell_types'][cell_name]['parameters']['Vinit'] + 10)} for _ in range(net_config['cell_types'][cell_name]['numerosity'])]
        nest.SetStatus(neuronal_populations[cell_name]['cell_ids'], dVinit)

    # assegnare gli ids generati da nest alle coordinate spaziali generati da BSB 
    try:
        neuronal_populations[cell_name]['cell_pos'] = network.get_placement_set(cell_name).load_positions()
    except Exception as e:
        print(f"{str(e)} population has not been positioned")
#%%-------------------------------------------COMPARE IDS---------------------
connection_models =list(net_config['connection_models'].keys())
for conn_model in connection_models:
    pre = net_config['connection_models'][conn_model]['pre']
    post = net_config['connection_models'][conn_model]['post']
    cs= network.get_connectivity_set(conn_model).load_connections()#.as_globals()
    data=cs.all()
    data_pre=data[0]
    data_post=data[1]
    index_pre = np.unique(data_pre[:,0])
    index_post = np.unique(data_post[:,0])
    
    print(pre)
    print(neuronal_populations[pre]['cell_ids'][0],neuronal_populations[pre]['cell_ids'][-1],len(neuronal_populations[pre]['cell_ids']))
    print(index_pre[0],index_pre[-1],len(index_pre))
    print(index_pre[0]+neuronal_populations[pre]['cell_ids'][0],index_pre[-1]+neuronal_populations[pre]['cell_ids'][0],len(index_pre))
    print('----------------------------')

    print(post)
    print(neuronal_populations[post]['cell_ids'][0],neuronal_populations[post]['cell_ids'][-1],len(neuronal_populations[post]['cell_ids']))
    print(index_post[0],index_post[-1],len(index_post))
    print(index_post[0]+neuronal_populations[post]['cell_ids'][0],index_post[-1]+neuronal_populations[post]['cell_ids'][0])
    print('----------------------------')

# %%
