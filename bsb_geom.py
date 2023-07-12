#%%-------------------------------------------IMPORT-----------------------------
'''Import'''
import json
import os
import sys

#**************BSB********************
from bsb.core import from_storage
network = from_storage("cerebellum_NODS.hdf5")

with open('demo_cerebellum.json', "r") as json_file:
    net_config = json.load(json_file)
pc_color  = net_config['colors']['purkinje_cell'][0]
grc_color  = net_config['colors']['granule_cell'][0]
nos_color  = net_config['colors']['nNOS'][0]


#%%**************NO DEPENDENCY**************
NO_dependency = True
#%%-------------------------------------------CREATE NETWORK---------------------
# estrarre numerosità da bsb: scaffold è sempre uguale, anche connettività, quindi può essere fatto tutto uno volta sola
MF_coord = network.get_placement_set("mossy_fibers").load_positions()
MF_num = len(MF_coord)
net_config['cell_num']['MF_num'] = MF_num
Glom_coord = network.get_placement_set("glomerulus").load_positions()
Glom_num = len(Glom_coord)
net_config['cell_num']['Glom_num'] = Glom_num
GR_coord = network.get_placement_set("granule_cell").load_positions()
GR_num = len(GR_coord)
net_config['cell_num']['GR_num'] = GR_num
PC_coord = network.get_placement_set("purkinje_cell").load_positions()
PC_num = len(PC_coord)
net_config['cell_num']['PC_num'] = PC_num
GO_coord = network.get_placement_set("golgi_cell").load_positions()
GO_num = len(GO_coord)
net_config['cell_num']['GO_num'] = GO_num
BC_coord = network.get_placement_set("basket_cell").load_positions()
BC_num = len(BC_coord)
net_config['cell_num']['BC_num'] = BC_num
SC_coord = network.get_placement_set("stellate_cell").load_positions()
SC_num = len(SC_coord)
net_config['cell_num']['SC_num'] = SC_num
IO_num = PC_num
net_config['cell_num']['IO_num'] = IO_num
DCN_num = round(PC_num/2)
net_config['cell_num']['DCN_num'] = DCN_num

with open('demo_cerebellum.json', "w") as json_file:
    json.dump(net_config,json_file,indent=4)
# definire modelli dei neuroni
# assegnare gli ids generati da nest alle coordinate spaziali generati da BSB 
# connettere ciclando su connessioni fatte da BSB
# verificare che con get.Connection si ritrovino cose sensate
# anche quando connetto PC, lo faccio per singola syn e assegno già coordinata nNOS e ev_point
