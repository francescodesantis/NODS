from bsb.core import from_storage
from bsb.morphologies import Morphology
import numpy as np
from scipy.spatial.distance import cdist
from bsb.plotting import plot_morphology
import plotly.graph_objects as go
from plotly.subplots import make_subplots

network = from_storage("cerebellum_NODS.hdf5")

'''# spot mossy fibres 
pos_mf =network.get_placement_set("mossy_fibers").load_positions()
spot_mf=[]
pos_spot_mf =[]
for n, position in enumerate(pos_mf):
   if ((30 < position[0] < 80 ) and (30 < position[2] < 40 )):
        spot_mf.append(n)
        pos_spot_mf.append(position)
print("num mossy: ", len(spot_mf))

#Get mossy_fibers_to_glomerulus connectivity set and select glomeruli connected to the selected mossy fibers
cs= network.get_connectivity_set('mossy_fibers_to_glomerulus').load_connections().as_globals()
data=cs.all()
data_mf = data[0]
data_spot_mf = data_mf[np.in1d(data_mf[:,0],spot_mf)]
data_glom = data[1]
data_spot_glom = data_glom[np.in1d(data_mf[:, 0], spot_mf)]
spot_glom = data_spot_glom[:,0]'''

# spot glomeruli
pos_glom =network.get_placement_set("glomerulus").load_positions()
spot_glom=[]
pos_spot_glom =[]
for n, position in enumerate(pos_glom):
   if ((30 < position[0] < 80 ) and (30 < position[2] < 40 )):
        spot_glom.append(n)
        pos_spot_glom.append(position)
print("num glom: ", len(spot_glom))

#Get glomerulus_to_granule connectivity set and select granule connected to the selected glomeruli
cs= network.get_connectivity_set('glomerulus_to_granule').load_connections().as_globals()
data=cs.all()
data_glom = data[0]
data_spot_glom = data_glom[np.in1d(data_glom[:, 0],spot_glom)]
data_grc = data[1]
data_spot_grc = data_grc[np.in1d(data_glom[:, 0],spot_glom)]
spot_grc = data_spot_grc[:,0]

selected_pc = 10
print("selected_pc: ", selected_pc)
#Get parallel_fiber_to_purkinje connectivity set
cs= network.get_connectivity_set('parallel_fiber_to_purkinje').load_connections().as_globals()
data=cs.all()
data_pre = data[0]
data_post = data[1]

#Filter data by selected pc cell
mask_data_pre=[]
mask_data_post= (data_post[:,0]==selected_pc)
data_post= data_post[mask_data_post]
data_pre= data_pre[mask_data_post]

#Filted data by selected grcs
mask=np.full(len(data_pre),False, dtype=bool)
for i in spot_grc:
    curr= data_pre[:,0]==i
    mask=mask|curr
data_post=data_post[mask]
data_pre=data_pre[mask]

#Load and translate pc morpho
pos_pc =network.get_placement_set("purkinje_cell").load_positions()
morpho_pcs =network.get_placement_set("purkinje_cell").load_morphologies()
morpho_itr = morpho_pcs.iter_morphologies()
morphos = []
for m in morpho_itr:
    morphos.append(m)
morpho_pc = morphos[selected_pc]
morpho_pc.translate(pos_pc[selected_pc])

fig = go.Figure()
voxels_coords = morpho_pc.voxelize(50).as_spatial_coords(copy=True)
voxel_boxes = morpho_pc.voxelize(50).as_boxes()
min_corners = voxel_boxes[:,0:3]
max_corners = voxel_boxes[:,3:6]
plot_morphology(morpho_pc,fig=fig, show=False, color="black")
#Get coordinates of the synapses
synapses_pos = []
for d in data_post:
    #print(morpho_pc.branches[d[1]].points[d[2]])
    synapses_pos.append(morpho_pc.branches[d[1]].points[d[2]])
    #print(synapses_pos)
synapses_pos = np.array(synapses_pos)
if len(synapses_pos)!=0:
    xpos = synapses_pos[:,0]
    ypos = synapses_pos[:,2]
    zpos = synapses_pos[:,1] 
    #Plot synapses and pc morphology
    fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=3, color="purple")))

#Get coordinates of all grcs body
pos_grc =network.get_placement_set("granule_cell").load_positions()
pos_grc = np.array(pos_grc)
xpos = pos_grc[:,0]
ypos = pos_grc[:,2]
zpos = pos_grc[:,1]
# all grc in the volume
fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=2,color='red',opacity=0.01)))

# only selected grcs
spot_grc_pos = []
for i in spot_grc:
    spot_grc_pos.append(pos_grc[i])
spot_grc_pos = np.array(spot_grc_pos)
xpos = spot_grc_pos[:,0]
ypos = spot_grc_pos[:,2]
zpos = spot_grc_pos[:,1] 
fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=2,color='red')))

#Get coordinates of all pcs body 
pos_pc = np.array(pos_pc)
xpos = pos_pc[:,0]
ypos = pos_pc[:,2]
zpos = pos_pc[:,1] 
fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=7,color='black',opacity=0.1)))

#Get coordinates of selected glom
pos_glom =network.get_placement_set("glomerulus").load_positions()
spot_glom_pos = []
for i in spot_glom:
    spot_glom_pos.append(pos_glom[i])
spot_glom_pos = np.array(spot_glom_pos)
xpos = spot_glom_pos[:,0]
ypos = spot_glom_pos[:,2]
zpos = spot_glom_pos[:,1] 
fig.add_trace(go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=7,color='yellow')))

fig.show()