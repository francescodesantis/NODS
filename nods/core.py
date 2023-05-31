import numpy as np
import math as m
import pandas as pd
from scipy import spatial
import dill

class NODS:
    def __init__(self,model_parameters):
        self.tauCa   = model_parameters['production']['tauCa']
        self.tauNOS1 = model_parameters['production']['tauNOS1']
        self.tauNOS2 = model_parameters['production']['tauNOS2']
        self.A       = model_parameters['production']['A']
        self.B       = model_parameters['diffusion']['B']
        self.D       = model_parameters['diffusion']['D'] # diffusion coefficient [um^2/ms]
        self.I       = model_parameters['diffusion']['I'] # inactivation coefficient [1/ms] 

        self.ds      = model_parameters['diffusion']['ds']
        self.r_max   = model_parameters['diffusion']['r_max']
        self.distances = np.arange(-self.r_max, self.r_max+self.ds, self.ds)
        r_2    = self.distances**2
        self.Green_LUT = np.zeros((len(self.distances),2))
        self.Green_LUT[:, 0] = Green_function(0, r_2, self.D, self.I)
        self.Green_LUT[:, 1] = Green_function(1, r_2, self.D, self.I)

        self.dt         = model_parameters['simulation']['dt']
        self.time       = np.arange(model_parameters['simulation']['t_start'], model_parameters['simulation']['t_end'], self.dt)
        self.Calm2C_0   = model_parameters['simulation']['Calm2C_0']
        self.nNOS_0     = model_parameters['simulation']['nNOS_0']
        self.NO_p_0     = model_parameters['simulation']['NO_p_0']

    def init_geometry(self, nNOS_coordinates, ev_point_coordinates, source_ids, nos_ids = None, ev_point_ids = None, file_ev_points = None, file_nNOS = None):

        if not file_ev_points:
            if not ev_point_ids:
                ev_point_ids = np.arange(len(ev_point_coordinates))
            self.ev_points = {}    
            for index,ev_point_id in enumerate(ev_point_ids):
                self.ev_points[ev_point_id] = dict(x =  ev_point_coordinates[index,0],
                                                   y =  ev_point_coordinates[index,1],
                                                   z =  ev_point_coordinates[index,2])
            #self.ev_points.sort_values(by='evpoint_id')
        #TODO else: load da file
        if not file_nNOS:
            if not nos_ids:
                nos_ids = np.arange(len(nNOS_coordinates))
            self.all_nNOS = {}
            for index,nos_id in enumerate(nos_ids):
                self.all_nNOS[nos_id] = dict(source_id = source_ids[index],
                                            x =  nNOS_coordinates[index,0],
                                            y =  nNOS_coordinates[index,1],
                                            z =  nNOS_coordinates[index,2])    
                
            #self.all_nNOS = pd.DataFrame({'source_id':source_ids, 'nos_id':nos_ids, 'x': nNOS_coordinates[:,0], 'y': nNOS_coordinates[:,1], 'z': nNOS_coordinates[:,2]})
        #TODO else: load da file
        self.sort_sources()
        return

    def sort_sources(self, filename = None):
        """function to filter the sources of nNOS activation to be avaluated"""
        self.relative_dist = []
        self.source_to_eval = []
        # loop on the receiver
        for evpoint_id in self.ev_points:
            # loop on the sources
            ev_point_coordinates = np.array([self.ev_points[evpoint_id]['x'],self.ev_points[evpoint_id]['y'],self.ev_points[evpoint_id]['z']])
            for nos_id in self.all_nNOS:                    
                nNOS_coordinates = np.array([self.all_nNOS[nos_id]['x'],self.all_nNOS[nos_id]['y'],self.all_nNOS[nos_id]['z']])                    
                # distance evaluation
                d = spatial.distance.euclidean(nNOS_coordinates, ev_point_coordinates)
                # check on relevant distance value
                if d < self.r_max:
                    # lists update
                    source_id = self.all_nNOS[nos_id]['source_id']
                    self.source_to_eval.append(source_id)
                    self.relative_dist.append([np.int(source_id), np.int(nos_id), np.int(evpoint_id), d]) # 0: id_source, 1: id_nos, 2:id_evpoint, 3: relative_distance
        # elimination repetition of same source
        self.source_to_eval = np.unique(self.source_to_eval)

        return
    
    def init_simulation(self,simulation_file, store_sim=True):

        self.NO_from_source = {}
        for source_id in self.source_to_eval:
            self.NO_from_source[source_id] = dict(  Calm2C = self.Calm2C_0 ,
                                                    nNOS   = self.nNOS_0,
                                                    NO_produced_t0  = self.NO_p_0,
                                                    NO_diffused     = np.zeros((len(self.distances), len(self.time))),
                                                    u               = np.zeros_like(self.distances).astype(np.float64),
                                                    NO_diffused_tf  = 0
                                                )
            
        self.NO_in_ev_points = np.zeros((len(self.time),len(self.ev_points)))
        
        if store_sim:
            dill.dump(self, open(simulation_file, "wb"))
        return 
    
    def load_simulation(self,simulation_file):
        return dill.load(open(simulation_file, "rb"))
    
    def store_simulation(self,simulation_file):
        dill.dump(self, open(simulation_file, "wb"))
        return 
        
    def evaluate_diffusion(self,active_sources,t):
     
        for source_id in self.source_to_eval:

            if source_id in active_sources:
                spike = 1
            else:
                spike = 0

            nNOS, Calm2C, NO_produced_t1 = Production_function(self.dt,spike,self.NO_from_source[source_id]['Calm2C'],self.NO_from_source[source_id]['nNOS'],self.tauCa,self.tauNOS1,self.tauNOS2,self.A)
            u, NO = Diffusion_function(self.dt,self.NO_from_source[source_id]['u'],self.Green_LUT,self.NO_from_source[source_id]['NO_produced_t0'],NO_produced_t1, self.B)
            self.NO_from_source[source_id]['Calm2C'] = Calm2C
            self.NO_from_source[source_id]['nNOS'] = nNOS
            self.NO_from_source[source_id]['NO_produced_t0'] = NO_produced_t1
            self.NO_from_source[source_id]['u'] = u
            self.NO_from_source[source_id]['NO_diffused_tf'] = NO
            self.NO_from_source[source_id]['NO_diffused'][:,t] = NO

        for row in self.relative_dist:# 0:id_source, 1:id_nos, 2:id_evpoint, 3:relative_distance
            d = row[3]
            # check on distance value
            if d < 0.2: #TODO renderlo settabile fuori come limite
                d = 0.2 # position of nNOS wrt center of the point source            
            # sum the contribution of each source 
            self.NO_in_ev_points[t, row[2]] += self.NO_from_source[row[0]]['NO_diffused_tf'][round((d+self.r_max)/self.ds)]                                
        return

def Production_function(dt,Ca_spike,Calm2C_old,nNOS_old,tauCa,tauNOS1,tauNOS2,A):
    
    Calm2C = Calm2C_old + (((Calm2C_old/tauCa) + Ca_spike)*dt)
    nNOS = nNOS_old+((((1/tauNOS1)*((Calm2C)/((Calm2C)+1)))-(nNOS_old/tauNOS2))*dt)
    NO = nNOS*A

    return nNOS, Calm2C, NO

def Green_function(t,r_2,D,I):
    eps = 0.1
    if t == 0:
        t = eps

    a = 1 / (4 * m.pi * D * t)
    e1 = (-1 * r_2) / (4 * D * t)
    exp_diffusion = np.exp(e1)
    exp_inactivation = np.exp(-I * t)
    G = (m.pow(a, 3 / 2)) * exp_diffusion * exp_inactivation    

    return G

def Diffusion_function(dt,u0,Green_LUT,NO_produced_t0,NO_produced_t1, B):

    spacial_conv = np.convolve(Green_LUT[:,1], u0, 'same')    
    u = spacial_conv + (((Green_LUT[:,0]*NO_produced_t1) + (Green_LUT[:,1]*NO_produced_t0))*((dt)/2))    
    NO = u*B 

    return u, NO