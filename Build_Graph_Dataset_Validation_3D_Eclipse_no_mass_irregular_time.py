import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, LayerNorm, ReLU, LeakyReLU, ELU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph, SAGEConv, to_hetero
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_geometric.data import Data,Dataset, InMemoryDataset, download_url, HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


import os
import os.path as osp
import pickle
from pickle import dump,load
import numpy as np
import math, time
import pandas as pd
import time

from Data_Processing import Read_ADGPRS_Rate
from Data_Processing import Read_ADGPRS_Var

import matplotlib.pyplot as plt 

import copy

from tqdm import tqdm
import sys
from os import sep, remove, getcwd
from os.path import join, sep
from glob import glob
import re
from itertools import islice
import psutil


def check_well_name_perf_number(cell_idx,well_names,num_inj,num_perf_per_well,inj_indices):
    count = 0
    for i in range(num_inj):
        for j in range(num_perf_per_well[i]):
            if inj_indices[count] == cell_idx:
                well_name = well_names[i]
                perf_number = j
                return well_name, perf_number
            count += 1
            
def input_sim_one_TS_helper(cell_idx,inj_indices,inj_well_index,p0,s0,d,phi,k,p1,s1,inj_mass_by_perf_t0,inj_mass_by_perf_t1,time_state_0,time_state_1,co2_plume_threshold = 0.1): # inj_mass_by_perf is negative value when injecting
    if cell_idx in inj_indices:
        xi = torch.tensor([[p0,s0,k,time_state_1-time_state_0,d,phi,1,inj_well_index[cell_idx][0],inj_mass_by_perf_t0]], dtype=torch.float) 
    else:
        xi = torch.tensor([[p0,s0,k,time_state_1-time_state_0,d,phi,0,0,0]], dtype=torch.float) 
    yi = torch.tensor([[p1,s1,inj_mass_by_perf_t1]], dtype=torch.float) 
    
    ## temporary coding: co2_plume_threshold should be input from user
    if s0 > co2_plume_threshold:
        co2_plume_mark_i = torch.tensor([True], dtype=torch.bool)
    else:
        co2_plume_mark_i = torch.tensor([False], dtype=torch.bool)
        
    return xi,yi,co2_plume_mark_i

def calc_well_loss_weights_single_cell(XYZ,idx,inj_indices,max_distance):
    curr_weights = 0
    for inj in inj_indices:
        curr_dist =  ((XYZ[idx,0]-XYZ[inj,0])**2+(XYZ[idx,1]-XYZ[inj,1])**2+(XYZ[idx,2]-XYZ[inj,2])**2)**0.5
        curr_weights += 1-curr_dist/max_distance
    curr_weights = curr_weights/(len(inj_indices))
    return curr_weights

def generate_inj_mass_by_perf_t0_t1(i,inj_indices,well_names,num_inj,num_perf_per_well,num_perf_per_well_dict,inj_mass_by_perf_init,inj_mass_by_perf,t0,t1):
    if i in inj_indices:
        well_name, perf_number = check_well_name_perf_number(i,well_names,num_inj,num_perf_per_well,inj_indices)
        if t0 == -1:
            inj_mass_by_perf_i_t0 = inj_mass_by_perf_init/num_perf_per_well_dict[well_name]
        else:
            inj_mass_by_perf_i_t0 = inj_mass_by_perf[well_name][perf_number][t0]
        inj_mass_by_perf_i_t1 = inj_mass_by_perf[well_name][perf_number][t1]
    else:
        inj_mass_by_perf_i_t0 = 0
        inj_mass_by_perf_i_t1 = 0
    return inj_mass_by_perf_i_t0, inj_mass_by_perf_i_t1


def input_sim_one_TS(sim_data_i,edge_attr,edge_index,t0,t1,depth,perm_log,poro,
                     conn_l,conn_r,saturation_init,pressure_init,inj_mass_by_perf_init,well_loc,well_index,grids,
                    trans,XYZ,Nc,num_edges,is_prev_features,num_prev_features,is_co2_front_loss, threshold,
                    is_multi_training,num_multi_training,is_range_well_loss,max_distance,is_prev_features_new,num_prev_features_new): # designed for 2D now, needs to adjust for 3D
    # first, intialization to feature-label-edge list and well_locations to check
    start_time = time.time()
    x = torch.tensor([], dtype=torch.float)
    co2_plume_mark = torch.tensor([], dtype=torch.bool)
    y = torch.tensor([], dtype=torch.float)
    num_inj = len(well_loc)
    well_names = list(well_loc.keys())
    num_perf_per_well = []
    num_perf_per_well_dict = {}
    for well_name in well_names:
        num_perf_per_well.append(len(well_loc[well_name]))
        num_perf_per_well_dict[well_name] = len(well_loc[well_name])

    inj_indices = []
    inj_well_index = {}
    for i in range(num_inj): 
        for j in range(num_perf_per_well[i]):
            curr_perf_loc = well_loc[well_names[i]][j]
            perf_index = curr_perf_loc[0]+curr_perf_loc[1]*grids[0]+curr_perf_loc[2]*grids[0]*grids[1]
            inj_indices.append(perf_index)
            inj_well_index[perf_index] = well_index[well_names[i]][j]
    
    pressure = sim_data_i['pressure']
    gas_saturation = sim_data_i['gas_saturation']
    inj_mass_by_perf = sim_data_i['inj_mass_by_perf']
    time_state = sim_data_i['time_state']
    # second, input node features (including pressure, saturation, perm, BHP-0 for matrix blocks, one-hot 100-inj 010-prod,001-matr) and node labels (including next P and S)
    if t0 == -1: # this is initial step
        first_step_mark = torch.tensor([True], dtype=torch.bool)
        for i in range(Nc):
            inj_mass_by_perf_i_t0,inj_mass_by_perf_i_t1 = generate_inj_mass_by_perf_t0_t1(i,inj_indices,well_names,num_inj,num_perf_per_well,num_perf_per_well_dict,inj_mass_by_perf_init,inj_mass_by_perf,t0,t1)
            xi,yi,co2_plume_mark_i = input_sim_one_TS_helper(i,inj_indices,inj_well_index,pressure_init[i],saturation_init[i],depth[i],poro[i],perm_log[i],
                                          pressure[t1][i],gas_saturation[t1][i],inj_mass_by_perf_i_t0,inj_mass_by_perf_i_t1,0,time_state[t1])
            x = torch.cat((x, xi),dim=0)
            co2_plume_mark = torch.cat((co2_plume_mark, co2_plume_mark_i),dim=0)
            y = torch.cat((y, yi),dim=0)
    else:
        first_step_mark = torch.tensor([False], dtype=torch.bool)
        for i in range(Nc):
            inj_mass_by_perf_i_t0,inj_mass_by_perf_i_t1 = generate_inj_mass_by_perf_t0_t1(i,inj_indices,well_names,num_inj,num_perf_per_well,num_perf_per_well_dict,inj_mass_by_perf_init,inj_mass_by_perf,t0,t1)
            xi,yi,co2_plume_mark_i = input_sim_one_TS_helper(i,inj_indices,inj_well_index,pressure[t0][i],gas_saturation[t0][i],
                                            depth[i],poro[i],perm_log[i], pressure[t1][i],gas_saturation[t1][i],inj_mass_by_perf_i_t0,inj_mass_by_perf_i_t1,time_state[t0],time_state[t1])
            x = torch.cat((x, xi),dim=0)
            co2_plume_mark = torch.cat((co2_plume_mark, co2_plume_mark_i),dim=0)
            y = torch.cat((y, yi),dim=0)
    
#     log_memory_usage()
    
    # Additionally, we can mark for the waterfront and add future states and use range well loss (not just at wellloc but a range)    
    if is_co2_front_loss:
        co2_front_mark = torch.zeros(x.size()[0],dtype=torch.bool)
        for idx in range(num_edges):
            node_l = edge_index[0][idx]; node_r = edge_index[1][idx];
            sat_l = x[node_l][1]; sat_r = x[node_r][1]
            if torch.abs(sat_l-sat_r) >= threshold:
                co2_front_mark[node_l] = True
                co2_front_mark[node_r] = True
                
    t0_original = t0; t1_original = t1  
    
    
    if is_multi_training:
        future_state = torch.zeros([y.size()[0],y.size()[1],num_multi_training],dtype=torch.float)
        for idx in range(num_multi_training):
            y_next = torch.tensor([], dtype=torch.float)
            t0 += 1; t1 += 1
            for i in range(Nc):
                inj_mass_by_perf_i_t0,inj_mass_by_perf_i_t1 = generate_inj_mass_by_perf_t0_t1(i,inj_indices,well_names,num_inj,num_perf_per_well,num_perf_per_well_dict,inj_mass_by_perf_init,inj_mass_by_perf,t0,t1)
                _,yi,_ = input_sim_one_TS_helper(i,inj_indices,inj_well_index,pressure[t0][i],gas_saturation[t0][i],depth[i],poro[i],perm_log[i], pressure[t1][i],gas_saturation[t1][i],inj_mass_by_perf_i_t0,inj_mass_by_perf_i_t1,time_state[t0],time_state[t1])
                y_next = torch.cat((y_next, yi),dim=0)
            future_state[:,:,idx] = y_next
    
    t0 = t0_original; t1 = t1_original  
    
    graph_data = Data(x=x, co2_plume_mark = co2_plume_mark, edge_index=edge_index, edge_attr=edge_attr, co2_front_mark=co2_front_mark, future_state = future_state, first_step_mark=first_step_mark, y=y)
    graph_data = T.ToUndirected()(graph_data)
    
#     log_memory_usage()
    
#     print(graph_data)
    return graph_data

def Load_Eclipse_Sim_Data_Static(main_dir,Nc,num_edges):
    # finally, we load the transmissibility and coordinate XYZ data
    fileName = main_dir+'/Transmissibility.DAT'
    data = pd.read_csv(fileName, delim_whitespace=True, skiprows=[0,1], names=['IN','OUT','TRANS'], dtype={'IN': str,'OUT': str,'TRANS': float}) 
    trans = data['TRANS'].values[:num_edges]
    
    fileName = main_dir+'/XYZ.in'
    XYZ = np.zeros((Nc,3))
    data = pd.read_csv(fileName, delim_whitespace=True)#, skiprows=[1])
    XYZ[:,0] = data['X'].values
    XYZ[:,1] = data['Y'].values
    XYZ[:,2] = data['Z'].values
    
    # Because model is so large now, we should use a generic max_dist instead of computing comparing all cells!
    # since our current problem is structured model the max_dist in from cell_0 to cell_N
    max_distance = 0
    for j in range(Nc):
        curr_dist = ((XYZ[0,0]-XYZ[j,0])**2+(XYZ[0,1]-XYZ[j,1])**2+(XYZ[0,2]-XYZ[j,2])**2)**0.5
        if curr_dist > max_distance:
            max_distance = curr_dist
                    
    return trans,XYZ, max_distance

def prepare_edge_attr(num_edges,XYZ,trans,edge_index):
    edge_attr = torch.tensor([],dtype=torch.float)
    for idx in range(num_edges):
        displacement = torch.tensor(XYZ[edge_index[1][idx]] - XYZ[edge_index[0][idx]])
        transmissibility = torch.tensor(trans[idx])
        distance = (displacement[0]**2+displacement[1]**2+displacement[2]**2)**0.5
        edge_attr_i = torch.tensor([[transmissibility,displacement[0],displacement[1],displacement[2],distance]], dtype=torch.float)
        edge_attr = torch.cat((edge_attr,edge_attr_i),dim=0) 
    return edge_attr

def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def Read_Eclipse_Output_Block_helper(model_dims,keyword,num_cells,steps,skip_line,rsm_path):
    # Initialize the data structures for saturation and pressure
    data_object = np.zeros((num_cells, steps)) 
    cell_count = 0
    # loop through file every 6+steps lines (chunks of data) at a time
    last_entry=[]
    with open(rsm_path,'r') as fid:
        while True:
            next_n_lines = list(islice(fid,skip_line+steps))
            if not next_n_lines:
                break
            col_name=re.split(r'\s+',next_n_lines[2])
            col_name_clean = [i for i in col_name if i]
            # if col name contains BGSAT, get the index
            ind=[]
            if keyword in col_name_clean:
                ind=indices(col_name_clean, keyword)
                #if i==1499:
                #    last_entry=next_n_lines
                chunck_list = []
                for k in range(skip_line,steps+skip_line):
                    chunck_bits = re.split(r'\s+',next_n_lines[k])
                    chunch_bits_clean = [chunck for chunck in chunck_bits if chunck]
                    chunck_list.append(chunch_bits_clean)
                for n in range(len(ind)):
                    for j in range(steps):
                        index = ind[n]
                        data_object[cell_count][j]=chunck_list[j][index]
                    cell_count = cell_count + 1
    # reshape data into 4D array
    data_object_4d = data_object.reshape((model_dims[0],model_dims[1],model_dims[2],steps), order='F') 
    
    return data_object_4d

def Read_Eclipse_Output_Well_helper(keyword, rsm_path, well_names, num_perf_per_well, steps, skip_line):
    # Initialize the data structures for saturation and pressure
    num_perf_tot = 0
    for i in range(len(num_perf_per_well)):
        num_perf_tot += num_perf_per_well[i]

    data_object = np.zeros((num_perf_tot, steps)) 
    cell_count = 0
    # loop through file every 6+steps lines (chunks of data) at a time
    last_entry=[]
    with open(rsm_path,'r') as fid:
        while True:
            next_n_lines = list(islice(fid,skip_line+steps))
            if not next_n_lines:
                break
            col_name=re.split(r'\s+',next_n_lines[2])
            col_name_clean = [i for i in col_name if i]
            ind=[]
            if keyword in col_name_clean:
                ind=indices(col_name_clean, keyword)
                chunck_list = []
                for k in range(skip_line,steps+skip_line):
                    chunck_bits = re.split(r'\s+',next_n_lines[k])
                    chunch_bits_clean = [chunck for chunck in chunck_bits if chunck]
                    chunck_list.append(chunch_bits_clean)
                for n in range(len(ind)):
                    for j in range(steps):
                        index = ind[n]
                        data_object[cell_count][j]=chunck_list[j][index]
                    cell_count = cell_count + 1

    return data_object

def Read_Eclipse_Output_and_Calculations(config,main_dir,model_dims,well_names,num_perf_per_well):
    path = os.path.join(main_dir, 'config_{}'.format(config))
    rsm_path=join(path,'CO2_ECLIPSE.RSM')
    
    skip_line = 6 # this includes everything above the actual data

    # Initialize the time steps
    time_steps = []
    # Find data block and get the time steps
    fid=open(rsm_path,'r')
    for _ in range(skip_line):
        next(fid)
    for line in fid:
        line_bits = re.split(r'\s+', line)
        line_bits_clean = [i for i in line_bits if i]
        if line_bits_clean == []:
            break
        time_steps.append(float(line_bits_clean[0]))
    fid.close()
    
    # Get the number of time steps
    steps = len(time_steps)
    num_cells = model_dims[0]*model_dims[1]*model_dims[2]
    
    BPR = Read_Eclipse_Output_Block_helper(model_dims,'BPR',num_cells,steps,skip_line,rsm_path)
    BGSAT = Read_Eclipse_Output_Block_helper(model_dims,'BGSAT',num_cells,steps,skip_line,rsm_path)
    
    inj_surface_rate_by_perf = Read_Eclipse_Output_Well_helper('CGFR', rsm_path, well_names, num_perf_per_well, steps, skip_line)
    
    return time_steps, BPR, BGSAT, inj_surface_rate_by_perf

# write a helper function that can read raw data from config_i files and get a object sim_data with state variables
def Load_Eclipse_Sim_Data_Dynamic(config,main_dir,well_loc,well_index,model_dims):
    gas_surface_density = 1.868 # kg/m3 for CO2
    well_names = well_loc.keys()
    num_perf_per_well = []
    for well_name in well_names:
        num_perf_per_well.append(len(well_loc[well_name]))
        
    time_steps,pres,sat_gas,inj_surface_rate_by_perf = Read_Eclipse_Output_and_Calculations(config,main_dir,model_dims,well_names,num_perf_per_well) 
    
    inj_mass_by_perf = inj_surface_rate_by_perf * gas_surface_density
    
    # rearrange into a dictionary
    inj_mass_by_perf_dict = {}
    count = 0
    well_names = list(well_names)
    for i in range(len(well_names)):
        curr_well_name = well_names[i]
        for j in range(num_perf_per_well[i]):
            if curr_well_name in inj_mass_by_perf_dict:
                inj_mass_by_perf_dict[curr_well_name].append(inj_mass_by_perf[count])
            else:
                inj_mass_by_perf_dict[curr_well_name] = []
                inj_mass_by_perf_dict[curr_well_name].append(inj_mass_by_perf[count])
            count +=1
            
    Nt = len(time_steps)
    Nx,Ny,Nz = model_dims[0],model_dims[1],model_dims[2]
    #  time_state: day; pressure: bar; pss_pressure: bar; gas_saturation: 1; inj_mass_by_perf: kg/day
#     sim_data = {'time_state':time_steps, 'pressure':pres.reshape((Nx*Ny*Nz,Nt),order='F').transpose(), 'pss_pressure':pss_pressure.reshape((Nx*Ny*Nz),order='F'), 'gas_saturation':sat_gas.reshape((Nx*Ny*Nz,Nt),order='F').transpose(), 'inj_mass_by_perf':inj_mass_by_perf_dict}
    sim_data = {'time_state':time_steps, 'pressure':pres.reshape((Nx*Ny*Nz,Nt),order='F').transpose(),'gas_saturation':sat_gas.reshape((Nx*Ny*Nz,Nt),order='F').transpose(), 'inj_mass_by_perf':inj_mass_by_perf_dict}
    
    return sim_data

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory used: {memory_info.rss / (1024**2)} MB")

class MyOwnDataset(Dataset):
    def __init__(self,root,num_batch,num_sim_per_batch,nconfig,num_time_step,depth,perm_log,poro,pressure_init,saturation_init,inj_mass_by_perf_init,grids,Nc,num_edges,
                 is_prev_features=False, num_prev_features=1,is_co2_front_loss = False, threshold = 0.4, is_multi_training = False, num_multi_training = 1, is_range_well_loss = False, 
                  is_prev_features_new = False, num_prev_features_new = 0, finished_process_batch = 0, is_sp_pres=True, transform=None, pre_transform=None):
        self.num_batch = num_batch
        self.num_sim_per_batch = num_sim_per_batch
        self.nconfig = nconfig
        self.num_time_step = num_time_step
        self.depth = depth
        self.perm = perm_log
        self.poro = poro
        self.pressure_init = pressure_init
        self.saturation_init = saturation_init
        self.inj_mass_by_perf_init = inj_mass_by_perf_init
        self.grids = grids
        self.Nc = Nc
        self.num_edges = num_edges
        self.data_dict = {}
        self.is_prev_features = is_prev_features
        self.num_prev_features = num_prev_features
        self.is_co2_front_loss = is_co2_front_loss
        self.threshold = threshold
        self.is_multi_training = is_multi_training
        self.num_multi_training = num_multi_training
        self.is_range_well_loss=is_range_well_loss
        self.is_prev_features_new = is_prev_features_new
        self.num_prev_features_new = num_prev_features_new
        self.finished_process_batch=finished_process_batch
        self.is_sp_pres=is_sp_pres
        super().__init__(root,transform,pre_transform)
        
        
    @property
    def raw_file_names(self):
        file_names = []
        for i in range(self.num_batch):
            file_names.append('sim_data_batch_no.'+str(i)+'.p')       
        return file_names
    
    @property
    def processed_file_names(self):
        file_names = []
        for i in range(self.num_batch):
            file_names.append('data_'+str(i+1)+'.pt')     
        return file_names

    
    def process(self):
        # Check memory usage before and after loading/processing data
        conn_l = pickle.load(open(self.raw_dir+'/conn_l.p','rb'))
        conn_r = pickle.load(open(self.raw_dir+'/conn_r.p','rb'))
        well_locs = pickle.load(open(self.raw_dir+'/wellLoc.p','rb'))
        well_indexs = pickle.load(open(self.raw_dir+'/wellIndex.p','rb'))
        
        trans,XYZ,max_distance = Load_Eclipse_Sim_Data_Static(self.raw_dir,self.Nc,self.num_edges)
        combined_array = np.array([conn_l, conn_r])
        edge_index = torch.tensor(combined_array, dtype=torch.long)
        edge_attr = prepare_edge_attr(self.num_edges,XYZ,trans,edge_index)  
       
#         log_memory_usage()

        for i in tqdm(range(self.finished_process_batch,self.num_batch),position=0,leave=True):
            print("we are processing batch No.{}".format(i+1))
            graph_data_all = []
            for idx in tqdm(range(self.num_sim_per_batch),position=0,leave=True):
                start_time = time.time()
                config = i*self.num_sim_per_batch+idx
                curr_well_loc = well_locs[config]
                curr_well_index = well_indexs[config]
                sim_data_i = Load_Eclipse_Sim_Data_Dynamic(config,self.raw_dir,curr_well_loc,curr_well_index,self.grids)
                
#                 log_memory_usage()
                
                for t in tqdm(range(self.num_time_step),position=0,leave=True):     
                    graph_data = input_sim_one_TS(sim_data_i, edge_attr, edge_index, t-1, t, self.depth,self.perm, self.poro, conn_l, conn_r, self.saturation_init, self.pressure_init,self.inj_mass_by_perf_init,  
                                                  curr_well_loc, curr_well_index, self.grids, trans,XYZ, self.Nc, self.num_edges, 
                                                  self.is_prev_features, self.num_prev_features,self.is_co2_front_loss,self.threshold,
                                                 self.is_multi_training, self.num_multi_training,self.is_range_well_loss,max_distance,self.is_prev_features_new,self.num_prev_features_new) #well_loc is library starting from 1
                    
#                     log_memory_usage()

                    graph_data_all.append(graph_data)
            torch.save(graph_data_all, osp.join(self.processed_dir, 'data_{}.pt'.format(i+1)))
  
        
            
    def len(self):
        return self.num_batch*self.num_sim_per_batch*self.num_time_step
    
    def get(self,idx): # this idx should be between 0-499999 if 50 batches, 0-9999 if 1 batch # we also perm normalization here for graph data!
        batch_idx, sim_time_idx = divmod(idx,self.num_sim_per_batch*self.num_time_step)
        if batch_idx not in self.data_dict:
            dataset = torch.load(osp.join(self.processed_dir,'data_{}.pt'.format(batch_idx+1)))
            self.data_dict[batch_idx] = dataset
        else:
            dataset = self.data_dict[batch_idx]
        data = dataset[sim_time_idx]
        return data    
# Due to the memory and other practical issues we need to split the normalization process into several sub-processes    
#########################################################################################
def normalization_helper_step1(data,normalization):
    inputs = data['x']
    outputs = data['y']
    
    # node feature
    inputs[:,0] = (inputs[:,0]-normalization['pressure'][0])/normalization['pressure'][1]
    inputs[:,1] = (inputs[:,1]-normalization['saturation'][0])/normalization['saturation'][1]
    inputs[:,2] = (inputs[:,2]-normalization['permeability'][0])/normalization['permeability'][1]
    inputs[:,3] = (inputs[:,3]-normalization['time_step'][0])/normalization['time_step'][1]
#     inputs[:,3] = (inputs[:,3]-normalization['pss_pressure'][0])/normalization['pss_pressure'][1]
    inputs[:,4] = (inputs[:,4]-normalization['depth'][0])/normalization['depth'][1] 
    inputs[:,5] = (inputs[:,5]-normalization['poro'][0])/normalization['poro'][1] 
    inputs[:,7] = (inputs[:,7]-normalization['wi'][0])/normalization['wi'][1] 
    inputs[:,8] = (inputs[:,8]-normalization['inj_mass_by_perf'][0])/normalization['inj_mass_by_perf'][1]
    data['x'] = inputs  
    
    # node label/prediction
    outputs[:,0] = (outputs[:,0]-normalization['pressure'][0])/normalization['pressure'][1]
    outputs[:,1] = (outputs[:,1]-normalization['saturation'][0])/normalization['saturation'][1]
    outputs[:,2] = (outputs[:,2]-normalization['inj_mass_by_perf'][0])/normalization['inj_mass_by_perf'][1]

    data['y'] = outputs
    
    return data
    
def apply_normalization_step1_train(train_dataset,method,known_norm):
    # step 1. obtain the normalization statistics
    if method =='known':
        normalization = known_norm
    else:
        normalization = {} # this one is not valid option yet
        
    # step 2. apply normalization to deepcopy form of the original data
    train_dataset_norm = train_dataset#copy.deepcopy(train_dataset)
    
    for data in tqdm(train_dataset_norm, total = len(train_dataset_norm)):
        data = normalization_helper_step1(data,normalization)
        
    return train_dataset_norm,normalization
    
#########################################################################################
    
def apply_normalization_step1_val_test(dataset,normalization):
    dataset_norm = dataset#copy.deepcopy(dataset)
    for data in tqdm(dataset_norm, total = len(dataset_norm)):
        data = normalization_helper_step1(data,normalization)   
    return dataset_norm

##########################################################################################
def normalization_helper_step2(data,normalization,is_prev_features,num_prev_features,is_multi_training,num_multi_training,is_prev_features_new,num_prev_features_new):
    inputs_edge = data['edge_attr']
    
    # edge feature
    inputs_edge[:,0] = (torch.log(inputs_edge[:,0])-normalization['transmissibility'][0])/normalization['transmissibility'][1]
    inputs_edge[:,1] = (inputs_edge[:,1]-normalization['displacement_x'][0])/normalization['displacement_x'][1]
    inputs_edge[:,2] = (inputs_edge[:,2]-normalization['displacement_y'][0])/normalization['displacement_y'][1]
    inputs_edge[:,3] = (inputs_edge[:,3]-normalization['displacement_z'][0])/normalization['displacement_z'][1]
    inputs_edge[:,4] = (inputs_edge[:,4]-normalization['distance'][0])/normalization['distance'][1]
    data['edge_attr'] = inputs_edge
    
    if is_multi_training:
        outputs_next = data['future_state']
        for idx in range(num_multi_training):
            outputs_next[:,0,idx] = (outputs_next[:,0,idx]-normalization['pressure'][0])/normalization['pressure'][1]
            outputs_next[:,1,idx] = (outputs_next[:,1,idx]-normalization['saturation'][0])/normalization['saturation'][1] 
            outputs_next[:,2,idx] = (outputs_next[:,2,idx]-normalization['inj_mass_by_perf'][0])/normalization['inj_mass_by_perf'][1] 
        data['future_state'] = outputs_next
        
    if is_prev_features_new:
        outputs_prev = data['prev_state']
        for idx in range(num_prev_features_new):
            outputs_prev[:,0,idx] = (outputs_prev[:,0,idx]-normalization['pressure'][0])/normalization['pressure'][1]
            outputs_prev[:,1,idx] = (outputs_prev[:,1,idx]-normalization['saturation'][0])/normalization['saturation'][1]           
            outputs_prev[:,2,idx] = (outputs_prev[:,2,idx]-normalization['inj_mass_by_perf'][0])/normalization['inj_mass_by_perf'][1]           
        data['prev_state'] = outputs_prev
    
    return data
    
def apply_normalization_step2(dataset,normalization,
                 is_prev_features=False, num_prev_features=1, is_multi_training = False, num_multi_training = 1,is_prev_features_new = False, num_prev_features_new = 1):
        
    dataset_norm = dataset#copy.deepcopy(dataset)
    
    for data in tqdm(dataset_norm, total = len(dataset_norm)):
        data = normalization_helper_step2(data,normalization,is_prev_features,num_prev_features,is_multi_training,num_multi_training,is_prev_features_new,num_prev_features_new)
        
    return dataset_norm



