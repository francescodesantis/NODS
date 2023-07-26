from bsb.core import from_storage
from bsb.morphologies import Morphology
import numpy as np
from scipy.spatial.distance import cdist
from bsb.plotting import plot_morphology
import plotly.graph_objects as go
from plotly.subplots import make_subplots

network = from_storage("cerebellum_NODS.hdf5")

# for ps in network.get_placement_sets():
#     print("numb of ", ps.tag, ": ", len(ps))

# spot GrCs
pos_grc =network.get_placement_set("granule_cell").load_positions()
spot_grc=[]
pos_spot_grc=[]
for n, position in enumerate(pos_grc):
   if ((30 < position[0] < 80 ) and (30 < position[2] < 80 ) and (280 < position[1] < 300 )):
        spot_grc.append(n)
        pos_spot_grc.append(position)
print("num grcs: ", len(spot_grc))

dist=[]
pos_pc =network.get_placement_set("purkinje_cell").load_positions()
for n, position in enumerate(pos_pc):
    p_a = np.array([position[0], position[2]])
    p_a = p_a.reshape(1, -1)
    p_b = np.array([70, 70])
    p_b = p_b.reshape(1, -1)
    dist.append([n,cdist(p_a, p_b, 'euclidean')])

selected_pc = min(dist, key=lambda x: x[1])[0]
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

#print(data_pre[:,0])

#Load and translate pc morpho
morpho_pcs =network.get_placement_set("purkinje_cell").load_morphologies()
morpho_itr = morpho_pcs.iter_morphologies()
morphos = []
for m in morpho_itr:
    morphos.append(m)
morpho_pc = morphos[selected_pc]
morpho_pc.translate(pos_pc[selected_pc])
#print(pos_pc[selected_pc])

#Load and translate grc morpho
morpho_grcs =network.get_placement_set("granule_cell").load_morphologies()
morpho_itr = morpho_grcs.iter_morphologies()
morphos_grcs= []
for m in morpho_itr:
    morphos_grcs.append(m)
morpho_grc = morphos_grcs[spot_grc[0]]

#Get coordinates of the synapses
synapses_pos = []
for d in data_post:
    #print(morpho_pc.branches[d[1]].points[d[2]])
    synapses_pos.append(morpho_pc.branches[d[1]].points[d[2]])
    #print(synapses_pos)
synapses_pos = np.array(synapses_pos)
xpos = synapses_pos[:,0]
#We need to swap y and z because of the choice of the axes in bsb.plotting
ypos = synapses_pos[:,2]
zpos = synapses_pos[:,1] 

#Plot synapses and pc morphology
fig = go.Figure(data=[go.Scatter3d(x=xpos, y=ypos, z=zpos, mode='markers', marker=dict(size=5))])
print("num pf synapses on the selected pc: ", len(xpos))


voxels_coords = morpho_pc.voxelize(50).as_spatial_coords(copy=True)
#fig=go.Figure(data=[go.Scatter3d(x=voxels_coords[:,0], y=voxels_coords[:,2], z=voxels_coords[:,1], mode='markers', marker=dict(size=6))])
voxel_boxes = morpho_pc.voxelize(50).as_boxes()
min_corners = voxel_boxes[:,0:3]
max_corners = voxel_boxes[:,3:6]
#fig.add_trace(go.Scatter3d(x=min_corners, y=min_corners, z=min_corners, mode='markers', marker=dict(size=6)))

plot_morphology(morpho_pc,fig=fig, show=False, color="black")
#for i in data_pre[:,0]:
for i in spot_grc:
    morpho_grc.translate(pos_grc[i])
    plot_morphology(morpho_grc,fig=fig, show=False, color="red")
    morpho_grc.translate(-pos_grc[i])

fig.show()

'''
in bsb/ bsb / plotting.py  / plot_morphology

    #for branch in morphology.branches:
        
        if "pf_targets" in branch.list_labels():
            color="blue"
        else:
            color="black"
        
        #traces.append(get_branch_trace(branch, offset, color=color, width=width))
    #for trace in traces:
'''