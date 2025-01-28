import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential as Seq, Linear, LayerNorm, ReLU, LeakyReLU, ELU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph, SAGEConv, to_hetero
from torch_geometric.utils import add_self_loops, degree, to_undirected
from torch_geometric.data import Data,Dataset, InMemoryDataset, download_url, HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler

import os
import os.path as osp
import pickle
from pickle import dump,load
import numpy as np
import math, time
import pandas as pd

import matplotlib.pyplot as plt 
from ast import literal_eval
import copy

import sys
sys.path.append('/home/groups/lou/hytang/SIMGNN_Modularization')
from Data_Processing import Build_Graph_Dataset_Validation_3D_Eclipse_no_mass_irregular_time as Build_Graph_Dataset
from Model import Model_new as Model
from Model import Training_satGNN_ccus_no_mass as Training
from Visualization import Rate_Curve_Visualization


def read_well_locations_by_perforation_and_wellIndex(file_path):
    wellLoc = {}
    wellIndex = {}
    start_reading = False

    with open(file_path, 'r') as file:
        for line in file:
            # Check for the start of the COMPDAT section
            if "COMPDAT" in line:
                start_reading = True
                continue

            # Process lines within the COMPDAT section
            if start_reading:
                if line == '/\n':
                    break
                parts = line.split()
                if parts[0].startswith("INJ"):  # Check if the line contains injector data
                    well_name = parts[0]
                    x, y, z = int(parts[1])-1, int(parts[2])-1, int(parts[3])-1 # since eclipse is 1-based location but GNN is 0-based location
                    WI = float(parts[7])
                    if well_name not in wellLoc:
                        wellLoc[well_name] = []
                        wellIndex[well_name] = []
                    wellLoc[well_name].append([x, y, z])
                    wellIndex[well_name].append([WI])

    return wellLoc, wellIndex

def apply_normalization_graph_data_init(graph_data_init,normalization):
    inputs = graph_data_init['x']
    inputs_edge = graph_data_init['edge_attr']

    # node feature
    inputs[:,0] = (inputs[:,0]-normalization['pressure'][0])/normalization['pressure'][1]
    inputs[:,1] = (inputs[:,1]-normalization['saturation'][0])/normalization['saturation'][1]
    inputs[:,2] = (inputs[:,2]-normalization['permeability'][0])/normalization['permeability'][1]
    inputs[:,3] = (inputs[:,3]-normalization['time_step'][0])/normalization['time_step'][1]
    inputs[:,4] = (inputs[:,4]-normalization['depth'][0])/normalization['depth'][1] 
    inputs[:,5] = (inputs[:,5]-normalization['poro'][0])/normalization['poro'][1] 
    inputs[:,7] = (inputs[:,7]-normalization['wi'][0])/normalization['wi'][1] 
    inputs[:,8] = (inputs[:,8]-normalization['inj_mass_by_perf'][0])/normalization['inj_mass_by_perf'][1]

    graph_data_init['x'] = inputs
    
    # edge feature
    inputs_edge[:,0] = (torch.log(inputs_edge[:,0])-normalization['transmissibility'][0])/normalization['transmissibility'][1]
    inputs_edge[:,1] = (inputs_edge[:,1]-normalization['displacement_x'][0])/normalization['displacement_x'][1]
    inputs_edge[:,2] = (inputs_edge[:,2]-normalization['displacement_y'][0])/normalization['displacement_y'][1]
    inputs_edge[:,3] = (inputs_edge[:,3]-normalization['displacement_z'][0])/normalization['displacement_z'][1]
    inputs_edge[:,4] = (inputs_edge[:,4]-normalization['distance'][0])/normalization['distance'][1]
    graph_data_init['edge_attr'] = inputs_edge
    
    return graph_data_init

def post_process_surrogate_output_reverse_normalization(pred_record,num_ts,normalization):
    P = []
    Sw = []
    inj_mass = []
    for i in range(num_ts):
        pred = pred_record[i]
        p_norm = pred[:,0]
        sw_norm = pred[:,1]
        inj_norm = pred[:,2]
        p = p_norm*normalization['pressure'][1]+normalization['pressure'][0]
        sw = sw_norm*normalization['saturation'][1]+normalization['saturation'][0]
        inj = inj_norm*normalization['inj_mass_by_perf'][1]+normalization['inj_mass_by_perf'][0]
        P.append(p.to('cpu').numpy())
        Sw.append(sw.to('cpu').numpy())
        inj_mass.append(inj.to('cpu').numpy())
    P = np.array(P)
    Sw = np.array(Sw)
    inj_mass = np.array(inj_mass)
    return P, Sw, inj_mass

def calculate_footprint(saturation_all, threshold=0.05):
    num_reals, nt, nx, ny, nz = saturation_all.shape
    footprint = np.zeros((num_reals, nt))

    for r in range(num_reals):
        for t in range(nt):
            # Get the saturation data for the current realization and time step
            saturation = saturation_all[r, t]

            # Thresholding: Filter out all cells above the threshold
            thresholded_saturation = np.where(saturation > threshold, saturation, 0)

            # Project everything in z direction into the first layer
            projected_saturation = np.max(thresholded_saturation, axis=2)

            # Find the bounding box for the projected plume shape
            x_coords, y_coords = np.where(projected_saturation > threshold)
            if len(x_coords) == 0 or len(y_coords) == 0:
#                 total_storage = 1  # To avoid division by zero, consider it as 1
                footprint = -1
            else:
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                # Calculate the area of the bounding box
                bounding_box_area = (x_max - x_min + 1) * (y_max - y_min + 1)
                # Calculate the total storage volume within the bounding box
                total_storage = bounding_box_area * nz
            footprint[r, t] = total_storage/(nx*ny*nz)

    return footprint

def process_features(features, max_len):
    repeated_features = []
    
    # Repeat the features until max_len is reached
    for i in range(max_len):
        repeated_features.append(features[i % features.shape[0]])
    
    # Convert the list to a numpy array
    repeated_features = np.array(repeated_features)

    return repeated_features

# Define the improved MLP model with residual connections and batch normalization
class ImprovedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, dropout_rate=0.3):
        super(ImprovedMLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size)])
        
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.residual = nn.Identity() if input_size == hidden_size else nn.Linear(input_size, hidden_size)

    def forward(self, x):
        residual = self.residual(x)
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            x = self.dropout(self.relu(bn(layer(x) + residual)))
            residual = x
        x = self.output_layer(x)
        return x

    
def load_models_and_scalers(save_dir, input_size, hidden_size, output_size, num_hidden_layers):
    # Load scalers
    with open(os.path.join(save_dir, 'training_BHP_predictor_MLP9_new_features_enhanced_first_step_3times_scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
        
    # Load models
    loaded_models = []
    model = ImprovedMLP(input_size, hidden_size, output_size, num_hidden_layers)
    model.load_state_dict(torch.load(os.path.join(save_dir, f'training_BHP_predictor_MLP9_new_features_enhanced_first_step_3times_model.pth')))
    model.eval()
    loaded_models.append(model)
    
    return loaded_models, scalers['input_scaler'], scalers['target_scaler']