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
import feval_helper as helper
import Build_Graph_Dataset_Validation_3D_Eclipse_no_mass_irregular_time as Build_Graph_Dataset
import Model_new as Model

def main(argv):
    infile = sys.argv[1] # decision variables each variable is own line
    outfile = sys.argv[2] # OBJ and constraints
    
    # step 1. load saturation gnn model
    satfile = '/python_resnet_small_dataset_satgnn_mass_flux_stage1_trial4_input.txt'

    with open(satfile) as f:
        mainlist = [list(literal_eval(line)) for line in f]
    config = {
          'num_hidden_layers':mainlist[0],
          'hidden_sizes': mainlist[1],
          'num_message_passing_steps':mainlist[2],
          'latent_sizes':mainlist[3],
          'noise':mainlist[4],
          'loss_weight':mainlist[5],
          'learning_rate':mainlist[6],
          'aggr_choice':mainlist[7],
          'message_choice':mainlist[8],
          'processor_choice':mainlist[9],
          'activation_type':mainlist[10],
          'group_norm_choice':mainlist[11]
    }
    is_waterfront_loss, threshold, gamma = mainlist[12][0],mainlist[12][1],mainlist[12][2]
    is_multi_training,num_multi_training,multi_training_weight = mainlist[13][0],mainlist[13][1],mainlist[13][2]
    is_range_well_loss = mainlist[14][0]
    is_prev_features_new, num_prev_features_new = mainlist[15][0],mainlist[15][1]
    num_epochs = mainlist[16][0]
    num_epochs_loaded = mainlist[17]
    model_name_base = mainlist[18][0]
    choice_dataloader = mainlist[19][0]
    batch_size_train_val_test = mainlist[20]
    is_load_opt = mainlist[21][0]#False#True#
    is_mass_flux = mainlist[22][0]
    first_step_loss_weight = mainlist[23][0]
    is_co2_plume_loss, co2_plume_cell_loss_ratio = mainlist[24][0],mainlist[24][1]

    # basic parameters for handling data
    num_batch = 30
    num_sim_per_batch = 10
    nconfig = num_batch*num_sim_per_batch
    num_time_step = 10
    num_ts = num_time_step
    num_time_step_tot = 12
    finished_process_batch = 0
    nstep = nconfig*num_time_step
    ##################################
    is_prev_features = False#False
    num_prev_features = 1
    ##################################
    oak_address = os.getcwd()
    # load dynamic data for each simulation run
    root = os.getcwd()
    raw_dir = root #+ 'raw'

    # generate connectivity list
    grids = [82,82,20]
    gridSize = [106,106,6.1]
    direction = [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]
    depth = 1524
    num_inj = 4
    num_prod = 0

    nx,ny,nz = grids[0],grids[1],grids[2]
    dx,dy,dz = gridSize[0],gridSize[1],gridSize[2]
    volume = dx*dy*dz
    num_cell = nx*ny*nz
    # load static data
    num_edges = 393436
    Nc = grids[0]*grids[1]*grids[2]


    fileName = raw_dir+"/PERM.DAT"
    size_x = grids[0]
    size_y = grids[1]
    size_z = grids[2]

    perm = np.zeros(size_x*size_y*size_z)
    perm_log = np.zeros(size_x*size_y*size_z)

    with open(fileName) as f:
        lines = f.readlines()
    for i in range(len(lines)-2):
        perm[i] = float(lines[i+1])
        perm_log[i] = math.log(perm[i])

    ###### initialization ######
    # in 3D model initial pressure is only specified for first layer, other layers need calculation to find out
    initial_pres_first_layer = 156.227 # bar, read from PRT file, consistent for the problem
    initial_pres_last_layer = 178.127 # bar, read from PRT file, consistent for the problem
    pressure_grad = (initial_pres_last_layer-initial_pres_first_layer)/gridSize[2]/(grids[2]-1)
    datum_depth = 1524 # m, from EQUIL of input file
    pres_at_datum_depth = 155.132 # bar, from EQUIL of input file

    initial_pres=np.zeros(Nc) 
    initial_gas_sat=np.zeros(Nc) # no CO2 at begining
    inj_mass_by_perf_init = -2739431/2 # this is the kg/day that to reach 1 mton/year

    # another way to do this is directly using the initial pressure from PRT file
    init_pres_layer = [155.435, 156.042, 156.649, 157.256, 157.863, 158.470, 159.077, 159.685, 160.292, 160.899,  161.506, 162.113, 162.720, 163.327, 163.934, 164.541, 165.149, 165.756, 166.363, 166.970]
    initial_pres_direct=np.zeros((grids[0],grids[1],grids[2]))
    for k in range(grids[2]):
        curr_depth = datum_depth + (k+.5)*gridSize[2]
        curr_pres = init_pres_layer[k]
        for j in range(grids[1]):
            for i in range(grids[0]):
                initial_pres_direct[i,j,k] = curr_pres

    final_pick = initial_pres_direct #initial_pres_calc#
    initial_pres = final_pick.reshape(Nc,order='F') # here only by 'F' the reshape would be column-major

     # finally, let's load the pore volume data
    fileName = raw_dir+'/Porosity.DAT'
    poro = np.zeros(size_x*size_y*size_z)

    with open(fileName) as f:
        lines = f.readlines()
    for i in range(len(lines)-2):
        poro[i] = float(lines[i+1])

    fileName = raw_dir+'/Depth.DAT'
    depth = np.zeros(size_x*size_y*size_z)

    with open(fileName) as f:
        lines = f.readlines()
    for i in range(len(lines)-2):
        depth[i] = float(lines[i+1]) 

    # first ele is min, second ele is max-min
    known_norm = {'pressure': [torch.tensor(155.435), torch.tensor(186.61111000000002)],
                     'saturation': [torch.tensor(0.), torch.tensor(1.0)],
                     'time_step': [torch.tensor(730.5000), torch.tensor(1.0000e-05)],
                     'wi': [torch.tensor(0.), torch.tensor(8210.48141)],
                     'inj_mass_by_perf': [torch.tensor(-1324677.25), torch.tensor(1324989.39601)],
                     'permeability': [torch.tensor(-4.6052), torch.tensor(13.8155)],              
                     'depth': [torch.tensor(1524.), torch.tensor(115.9000)],              
                     'poro': [torch.tensor(0.0500), torch.tensor(0.3200)],              
                     'transmissibility': [torch.tensor(-9.5850), torch.tensor(17.2232)],
                     'displacement_x': [torch.tensor(0.), torch.tensor(106.)],
                     'displacement_y': [torch.tensor(0.), torch.tensor(106.)],
                     'displacement_z': [torch.tensor(-6.1000), torch.tensor(6.1000)],
                     'distance': [torch.tensor(6.1000), torch.tensor(99.9000)]}


    dataset = Build_Graph_Dataset.MyOwnDataset(root, num_batch, num_sim_per_batch, nconfig, num_time_step, depth, perm_log,
                                           poro, initial_pres, initial_gas_sat, inj_mass_by_perf_init, grids,Nc,num_edges,
                                           is_prev_features,num_prev_features,is_waterfront_loss, threshold,
                                          is_multi_training, num_multi_training,is_range_well_loss,
                                           is_prev_features_new,num_prev_features_new,finished_process_batch)

    input_size = dataset[0].x.shape[1]
    edge_input_size = dataset[0].edge_attr.shape[1]
    output_size =  dataset[0].y.shape[1]


    # for tunning hyperparameter purpose only!!!
    is_save = True#False#
    is_sat = True#False#
    is_res_net = True#False#
    is_pretrained_pres_gnn = False
    pres_gnn_model = None
    sat_model_list = []
    finished_trial = []

    finished_trial = []
    # model_name_base ='Main_V3_presGNN_30X50_new_loss_trials_No.'
    model_path_base = '/home/groups/lou/hytang/SIMGNN_3d_structured_ccus_Illinois_final_no_pss/saved_models/satgnn/'

    if is_sat:
        index = 1
    else:
        index = 0

    save_folder = '/home/groups/lou/hytang/SIMGNN_checkpoint/3d_ccus_illinois_final_no_pss/satgnn/'
    choices = {
                   'hidden':len(config['hidden_sizes']),
                    'latent':len(config['latent_sizes']),
                   'message':len(config['num_message_passing_steps']),
                    'noise':len(config['noise']),
                    'weight':len(config['loss_weight']),
                    'learning_rate':len(config['learning_rate']),
                    'aggr_choice':len(config['aggr_choice']),
                    'message_choice':len(config['message_choice']),
                    'processor_choice':len(config['processor_choice']),
                    'activation_type':len(config['activation_type']),
                    'group_norm_choice':len(config['group_norm_choice']),
                  }


    min_loss = 1e5
    count = 0

    for i in range(choices['hidden']):
        for j in range(choices['message']):
            for k in range(choices['latent']):
                for l in range(choices['noise']):
                    for m in range(choices['weight']):
                        for n in range(choices['learning_rate']):
                            for o in range(choices['aggr_choice']):
                                for p in range(choices['message_choice']):
                                    for q in range(choices['processor_choice']):
                                        for r in range(choices['activation_type']):
                                            for s in range(choices['group_norm_choice']):
                                                count = count + 1
                                                model_name=model_name_base+str(count)
                                                model_path =model_path_base+model_name

                                                hidden_sizes = config['hidden_sizes'][i]
                                                num_hidden_layers = config['num_hidden_layers'][i]
                                                num_message_passing_steps = config['num_message_passing_steps'][j]
                                                latent_size = config['latent_sizes'][k]
                                                pressure_noise = [0,config['noise'][l][0]]
                                                saturation_noise = [0,config['noise'][l][1]]
                                                alpha = config['loss_weight'][m][0] # ctrl the weight of MAE loss
                                                beta = config['loss_weight'][m][1] # ctrl the weight of well loss
                                                aggr_choice = config['aggr_choice'][o]
                                                message_choice = config['message_choice'][p]
                                                processor_choice = config['processor_choice'][q]
                                                activation_type = config['activation_type'][r]
                                                group_norm_choice = config['group_norm_choice'][s]

                                                torch.cuda.empty_cache()
                                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                                model = Model.SIMGNN(input_size, edge_input_size,output_size, latent_size, hidden_sizes, num_hidden_layers, 
                                                                     num_message_passing_steps,aggr_choice,message_choice,is_res_net,processor_choice,activation_type,group_norm_choice).to(device)
                                                load_model = model
                                                load_model.load_state_dict(torch.load(model_path))
                                                load_model.eval()
                                                sat_model_list.append(load_model)    

    # step 2. load pressure gnn model
    presfile = '/python_resnet_middle_dataset_mass_flux_stage1_trial1_input.txt'

    with open(presfile) as f:
        mainlist = [list(literal_eval(line)) for line in f]
    config = {
          'num_hidden_layers':mainlist[0],
          'hidden_sizes': mainlist[1],
          'num_message_passing_steps':mainlist[2],
          'latent_sizes':mainlist[3],
          'noise':mainlist[4],
          'loss_weight':mainlist[5],
          'learning_rate':mainlist[6],
          'aggr_choice':mainlist[7],
          'message_choice':mainlist[8],
          'processor_choice':mainlist[9],
          'activation_type':mainlist[10],
          'group_norm_choice':mainlist[11]
    }
    is_waterfront_loss, threshold, gamma = mainlist[12][0],mainlist[12][1],mainlist[12][2]
    is_multi_training,num_multi_training,multi_training_weight = mainlist[13][0],mainlist[13][1],mainlist[13][2]
    is_range_well_loss = mainlist[14][0]
    is_prev_features_new, num_prev_features_new = mainlist[15][0],mainlist[15][1]
    num_epochs = mainlist[16][0]
    num_epochs_loaded = mainlist[17]
    model_name_base = mainlist[18][0]
    choice_dataloader = mainlist[19][0]
    batch_size_train_val_test = mainlist[20]
    is_load_opt = mainlist[21][0]#False#True#
    is_mass_flux = mainlist[22][0]
    first_step_loss_weight = mainlist[23][0]

    # for tunning hyperparameter purpose only!!!
    is_save = True#False#
    is_sat = False#True#
    is_res_net = True#False#
    pres_model_list = []
    finished_trial = []
    # model_name_base ='Main_V3_presGNN_30X50_new_loss_trials_No.'
    model_path_base = '/home/groups/lou/hytang/SIMGNN_3d_structured_ccus_Illinois_final_no_pss/saved_models/presgnn/'

    if is_sat:
        index = 1
    else:
        index = 0

    save_folder = '/home/groups/lou/hytang/SIMGNN_checkpoint/3d_ccus_illinois_final_no_pss/presgnn/'
    choices = {
                   'hidden':len(config['hidden_sizes']),
                    'latent':len(config['latent_sizes']),
                   'message':len(config['num_message_passing_steps']),
                    'noise':len(config['noise']),
                    'weight':len(config['loss_weight']),
                    'learning_rate':len(config['learning_rate']),
                    'aggr_choice':len(config['aggr_choice']),
                    'message_choice':len(config['message_choice']),
                    'processor_choice':len(config['processor_choice']),
                    'activation_type':len(config['activation_type']),
                    'group_norm_choice':len(config['group_norm_choice']),
                  }


    min_loss = 1e5
    count = 0

    for i in range(choices['hidden']):
        for j in range(choices['message']):
            for k in range(choices['latent']):
                for l in range(choices['noise']):
                    for m in range(choices['weight']):
                        for n in range(choices['learning_rate']):
                            for o in range(choices['aggr_choice']):
                                for p in range(choices['message_choice']):
                                    for q in range(choices['processor_choice']):
                                        for r in range(choices['activation_type']):
                                            for s in range(choices['group_norm_choice']):
                                                count = count + 1

                                                model_name=model_name_base+str(count)
                                                model_path =model_path_base+model_name

                                                hidden_sizes = config['hidden_sizes'][i]
                                                num_hidden_layers = config['num_hidden_layers'][i]
                                                num_message_passing_steps = config['num_message_passing_steps'][j]
                                                latent_size = config['latent_sizes'][k]
                                                pressure_noise = [0,config['noise'][l][0]]
                                                saturation_noise = [0,config['noise'][l][1]]
                                                alpha = config['loss_weight'][m][0] # ctrl the weight of MAE loss
                                                beta = config['loss_weight'][m][1] # ctrl the weight of well loss
                                                aggr_choice = config['aggr_choice'][o]
                                                message_choice = config['message_choice'][p]
                                                processor_choice = config['processor_choice'][q]
                                                activation_type = config['activation_type'][r]
                                                group_norm_choice = config['group_norm_choice'][s]

                                                torch.cuda.empty_cache()
                                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                                model = Model.SIMGNN(input_size, edge_input_size,output_size, latent_size, hidden_sizes, num_hidden_layers, 
                                                                     num_message_passing_steps,aggr_choice,message_choice,is_res_net,processor_choice,activation_type,group_norm_choice).to(device)
                                                load_model = model
                                                load_model.load_state_dict(torch.load(model_path))
                                                load_model.eval()

                                                pres_model_list.append(load_model)
                                                
    # step 3. prepare initial graph data
    # prepare x
    x = torch.tensor([], dtype=torch.float)
    time_state_0 = 0
    time_state_1 = 7.3050e+02
    file_path = 'base.sched'
    well_loc, well_index = helper.read_well_locations_by_perforation_and_wellIndex(file_path)
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
    ### how to get initial_pres,initial_gas_sat,perm_log[cell_idx],time_state_1-time_state_0,depth[cell_idx],poro[cell_idx],1,inj_well_index[cell_idx][0],inj_mass_by_perf_t0???
    for cell_idx in range(num_cell):
        if cell_idx in inj_indices:
            well_name, perf_number = Build_Graph_Dataset.check_well_name_perf_number(cell_idx,well_names,num_inj,num_perf_per_well,inj_indices)
            inj_mass_by_perf_t0 = inj_mass_by_perf_init/num_perf_per_well_dict[well_name]
            xi = torch.tensor([[initial_pres[cell_idx],initial_gas_sat[cell_idx],perm_log[cell_idx],time_state_1-time_state_0,depth[cell_idx],poro[cell_idx],1,inj_well_index[cell_idx][0],inj_mass_by_perf_t0]], dtype=torch.float) 
        else:
            xi = torch.tensor([[initial_pres[cell_idx],initial_gas_sat[cell_idx],perm_log[cell_idx],time_state_1-time_state_0,depth[cell_idx],poro[cell_idx],0,0,0]], dtype=torch.float)
        x = torch.cat((x, xi),dim=0)

    # prepare edge_index & edge_attr
    conn_l = pickle.load(open(raw_dir+'/conn_l.p','rb'))
    conn_r = pickle.load(open(raw_dir+'/conn_r.p','rb'))

    trans,XYZ,max_distance = Build_Graph_Dataset.Load_Eclipse_Sim_Data_Static(raw_dir,Nc,num_edges)
    combined_array = np.array([conn_l, conn_r])
    edge_index = torch.tensor(combined_array, dtype=torch.long)
    edge_attr = Build_Graph_Dataset.prepare_edge_attr(num_edges,XYZ,trans,edge_index) 

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph_data_init = T.ToUndirected()(graph_data)
    
    # step 4. normalize graph_data_init and perform rollout to get predictions and back normalization to return to original space
    normalization=known_norm
    graph_data_init_norm = helper.apply_normalization_graph_data_init(graph_data_init,known_norm)
    pres_model,sat_model = pres_model_list[0],sat_model_list[0]

    data = copy.deepcopy(graph_data_init_norm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_record = []
    for i in range(num_ts):
        pres_model.eval();sat_model.eval()
        torch.set_grad_enabled(False)
        data = data.to(device, non_blocking=True)
        pres_pred = pres_model(data)
        sat_pred = sat_model(data)
        pred_normal = pres_pred
        pred_normal[:,1:] = sat_pred[:,1:]
        data.x[:,0] = pred_normal[:,0]
        data.x[:,1] = pred_normal[:,1]
        data.x[:,8] = pred_normal[:,2]
        pred_record.append(pred_normal) #here this is normalized data not P and Sw directly!!!
        data = data.to('cpu')

    P, Sw, inj_mass = helper.post_process_surrogate_output_reverse_normalization(pred_record,num_ts,known_norm)
    
    # step 5. calculate footprint base on current outputs
    num_reals = 1
    x_dim = grids[0]
    y_dim = grids[1]
    z_dim = grids[2]
    test_batch_size = 1

    saturation_all = np.zeros((num_reals,num_ts,x_dim,y_dim,z_dim)) #(#reals,#timestep,#x_dim,#y_dim)
    for i in range(num_reals):
        for j in range(num_time_step):
            idx = i*num_time_step+j
            var_tensor = Sw[j]
            var = var_tensor.reshape((x_dim,y_dim,z_dim),order='F')
            saturation_all[i,j] = var  

    co2_footprint = helper.calculate_footprint(saturation_all)
    footprint = co2_footprint[0][-1]
    
    # step 6. BHP prediction base on current outputs
    num_test = 1
    nt = num_ts
    well_loc_test = [well_loc]
    # Initialize storage for features
    inj1_feature = []
    inj2_feature = []
    inj3_feature = []
    inj4_feature = []

    # Function to calculate 1D index from 3D coordinates
    def get_1d_index(x, y, z, nx=82, ny=82, nz=20):
        return z * (nx * ny) + y * nx + x

    # Extract features
    for i in range(nt):
        realization_idx = i // 10
        time_step_idx = i % 10

        tensor = np.array([P[i], Sw[i], inj_mass[i]]).reshape((-1,3))
        tensor = torch.tensor(tensor)
        if tensor.is_cuda:
            tensor = tensor.cpu()
        well_locs = well_loc_test[realization_idx]

        # Extract 1D indices for the well locations
        inj1_idices = []
        for j in range(len(well_locs['INJ1'])):
            inj1_idices.append(get_1d_index(*well_locs['INJ1'][j]))

        inj2_idices = []
        for j in range(len(well_locs['INJ2'])):
            inj2_idices.append(get_1d_index(*well_locs['INJ2'][j]))

        inj3_idices = []
        for j in range(len(well_locs['INJ3'])):
            inj3_idices.append(get_1d_index(*well_locs['INJ3'][j]))

        inj4_idices = []
        for j in range(len(well_locs['INJ4'])):
            inj4_idices.append(get_1d_index(*well_locs['INJ4'][j]))


        # Extract features for each well and append to corresponding lists
        inj1_feature.append(tensor[inj1_idices])
        inj2_feature.append(tensor[inj2_idices])
        inj3_feature.append(tensor[inj3_idices])
        inj4_feature.append(tensor[inj4_idices])

    # Convert lists to arrays for easier handling
    inj1_feature_np = [tensor.numpy() for tensor in inj1_feature]
    inj1_feature = np.stack(inj1_feature_np)
    inj2_feature_np = [tensor.numpy() for tensor in inj2_feature]
    inj2_feature = np.stack(inj2_feature_np)
    inj3_feature_np = [tensor.numpy() for tensor in inj3_feature]
    inj3_feature = np.stack(inj3_feature_np)
    inj4_feature_np = [tensor.numpy() for tensor in inj4_feature]
    inj4_feature = np.stack(inj4_feature_np)

    max_len = 18
    # Initialize BHP_features
    BHP_features = {
        'INJ1': [[] for _ in range(num_reals)],  # 150 realizations, each with an empty list for time steps
        'INJ2': [[] for _ in range(num_reals)],
        'INJ3': [[] for _ in range(num_reals)],
        'INJ4': [[] for _ in range(num_reals)]
    }

    # Process each feature set in inj1_feature
    for i, features in enumerate(inj1_feature):
        realization_idx = i // nt
        time_step_idx = i % nt

        processed_feature = helper.process_features(features,max_len).reshape(-1)

        BHP_features['INJ1'][realization_idx].append(processed_feature)

    # Process each feature set in inj1_feature
    for i, features in enumerate(inj2_feature):
        realization_idx = i // nt
        time_step_idx = i % nt

        processed_feature = helper.process_features(features,max_len).reshape(-1)

        BHP_features['INJ2'][realization_idx].append(processed_feature)

    # Process each feature set in inj1_feature
    for i, features in enumerate(inj3_feature):
        realization_idx = i // nt
        time_step_idx = i % nt

        processed_feature = helper.process_features(features,max_len).reshape(-1)

        BHP_features['INJ3'][realization_idx].append(processed_feature)

    # Process each feature set in inj1_feature
    for i, features in enumerate(inj4_feature):
        realization_idx = i // nt
        time_step_idx = i % nt

        processed_feature = helper.process_features(features,max_len).reshape(-1)

        BHP_features['INJ4'][realization_idx].append(processed_feature)
        
    # Directory to save models
    save_dir = "/home/groups/lou/hytang/SIMGNN_3d_structured_ccus_Illinois_final_no_pss/visualizations/saved_models"
    os.makedirs(save_dir, exist_ok=True)


    # Load your data
    BHP_features_loaded = BHP_features

    # Hyperparameters
    input_size = 3*18
    output_size = 1
    num_time_step = 10
    num_epochs = 3000
    learning_rate = 0.001
    max_lr = 0.01
    warmup_epochs = 100

    # Example of loading the models
    # Best found parameters from your previous runs
    hidden_size = 1024
    num_hidden_layers = 10

    loaded_models, input_scaler, target_scaler = helper.load_models_and_scalers(save_dir, input_size, hidden_size, output_size, num_time_step)

    # Prepare the data
    X = []

    for well in ['INJ1', 'INJ2', 'INJ3', 'INJ4']:
        X.extend(BHP_features_loaded[well])

    X = np.array(X, dtype=np.float32)

    # Ensure the data is in the correct format (num_samples, num_time_steps, num_features)
    X = X.reshape(-1, num_time_step, input_size)

    # Normalize the input features
    scaler = StandardScaler()
    X = X.reshape(-1, input_size)  # Flatten the time steps for normalization
    X = scaler.fit_transform(X)
    X = X.reshape(-1, num_time_step, input_size)  # Reshape back to the original shape

    # Flatten the data
    X_flattened = torch.tensor(X).view(-1, input_size)  # Shape: (4800, 6)

    # Plotting predictions vs actual values for the best model
    best_model = loaded_models[0]
    best_model.eval()
    with torch.no_grad():
        X_flattened = X_flattened.clone().detach()
        y_pred_all = best_model(X_flattened)

    y_pred_all = y_pred_all.reshape((-1,4,10,1))
    # Inverse transform the predictions and the actual values
    y_pred_all_original = target_scaler.inverse_transform(y_pred_all.reshape(-1, 1)).reshape(-1, 4, 10, 1)

    max_value = np.max(y_pred_all_original)
    
    # step 7. out-of-boundary CO2 prediction on current outputs
    # basically this is to calculate the saturation of each boundary block X their porosity = rudrm_value
        # and also to calculate the saturation of each inner blocks X their porosity + rudrm_value = fudrmtot_value
    poro_volume_multipliers = [276,76207] # the multipler for the edge cells and corner cells
    dx,dy,dz = 106,106,6.1
    volume = dx*dy*dz
    rudrm_value = 0
    fudrmtot_value = 0
    curr_sat = Sw[-1]
    is_rudrm = False
    # check saturation in each cell
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = get_1d_index(i,j,k)
                phi = poro[idx]
                sat = curr_sat[idx]
                if (i == 0 and j == 0) or (i == 81 and j == 0) or (i == 0 and j == 81) or (i == 81 and j == 81):
                    mult = poro_volume_multipliers[1]
                    is_rudrm = True
                elif i == 0 or j == 0 or i == 81 or j == 81:
                    mult = poro_volume_multipliers[0]
                    is_rudrm = True
                else:
                    mult = 1
                    is_rudrm = False

                    rudrm_value += is_rudrm*mult*volume*phi*sat
                    fudrmtot_value += mult*volume*phi*sat
                
    max_bhp = 276.3 #155.435*1.5 # 1.5 x of initial pressure at the top of storage aquifer

    constraint1 = max(0,max_value-max_bhp)/max_bhp
    constraint2 = rudrm_value / fudrmtot_value

    # print to file
    outFileName = 'objective.out'
    f= open(outFileName,"w+")
    f.close()

    try:
        f = open(outFileName, 'a')
        f.write(str(-footprint)) # the reason footprint is negative is we maximize the objective, but we want the footprint to be minimized instead
        f.write('\n')
        f.write(str(constraint1))
        f.write('\n')
        f.write(str(constraint2))
        f.close()
    except:
        print("Unable to append to "+ outFileName)
        
    return
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
