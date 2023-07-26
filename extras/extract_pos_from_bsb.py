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
hdf5_file = "cerebellum_NODS_smaller"
network = from_storage(hdf5_file+".hdf5")

with open('demo_cerebellum.json', "r") as json_file:
    net_config = json.load(json_file)

network_geom_file = 'geom_'+hdf5_file
network_connectivity_file = 'conn_'+hdf5_file
#%%
cell_types = list(net_config['cell_types'].keys())
neuronal_populations = {cell_name:{'cell_ids':[],'cell_pos':[]} for cell_name in cell_types}
for cell_name in cell_types:
    neuronal_populations[cell_name]['cell_ids'] = [i for i in range(net_config['cell_types'][cell_name]['numerosity'])]
    try:
        neuronal_populations[cell_name]['cell_pos'] = network.get_placement_set(cell_name).load_positions()        
    except Exception as e:
        print(f"{str(e)} population has not been positioned")
        neuronal_populations[cell_name]['cell_pos'] = 0

dill.dump(neuronal_populations,open(network_geom_file, "wb"))
   

# %%
geom = dill.load(open(network_geom_file, "rb"))
geom['io_cell']['cell_ids']

# %%
connection_models = list(net_config['connection_models'].keys())
connectivity_matrices = {connection:{'id_pre':[],'id_post':[]} for connection in connection_models}

for connection in connection_models:
    pre = net_config['connection_models'][connection]['pre']
    if pre!= 'io_cell':
        cs= network.get_connectivity_set(connection).load_connections()#.as_globals()
        data=cs.all()
        data_pre = data[0]
        connectivity_matrices[connection]['id_pre']=data_pre[:,0]
        data_post = data[1]
        connectivity_matrices[connection]['id_post']=data_post[:,0]

dill.dump(connectivity_matrices,open(network_connectivity_file, "wb"))

# %%
conn = dill.load(open(network_connectivity_file, "rb"))
conn['glomerulus_to_granule']['id_pre']
# %%
