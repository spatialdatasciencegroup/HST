import pickle
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bisect import bisect_left
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from quadtreev2 import Point, Node, QTree
from matplotlib import gridspec
from quadtreev2 import find_children
from utils.bandLayers import *
import os
import random

from time import perf_counter

torch.set_default_dtype(torch.float32)


TIME_WINDOW = 700
longest_window = 1000
subpoint_num = 45
number_selection = 1
regression  = True
LEARNING_RATE = 5e-4
BATCH_SIZE = 256
weight_decay = 5e-3
device = torch.device('cuda:{}'.format(2))

val_save_folder = 'trained_model/32/'
if not os.path.exists(val_save_folder):
    os.mkdir(val_save_folder)

sparse_homepath = '/blue/zhe.jiang/whe2/band2/1000_50/'
if not os.path.exists(sparse_homepath):
    os.mkdir(sparse_homepath)
    
epison = torch.tensor(1e-4)
prior_uq = 6

opt = dict()
opt['src_vocab_size'] = 8
opt['trg_vocab_size'] = 8
opt['src_pad_idx'] = -10
opt['trg_pad_idx'] = -10
opt['embed_size'] = 128
opt['num_layers'] = 1
opt['forward_expansion'] = 1
opt['heads'] = 1
opt['dropout'] = 0.2
opt['device'] = device
opt['max_length'] = 10

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0
    if pos == len(myList):
        return len(myList) - 1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos - 1
def calCord(latarray, longarray):
    xaxis = []
    yaxis = []
    for i in range(latarray.shape[0]): 
        lat = latarray[i]
        long = longarray[i]
        
        latMid = (lat+minlat )/2.0


        m_per_deg_lat = (111132.954 - 559.822 * math.cos( 2.0 * latMid ) + 1.175 * math.cos( 4.0 * latMid)) / 1e5
        m_per_deg_lon = ((3.14159265359/180 ) * 6367449 * math.cos ( latMid ))/1e5

        deltaLat = math.fabs(lat - minlat)
        deltaLon = math.fabs(long - minlong)
        
        xaxis.append(deltaLat * m_per_deg_lat)
        yaxis.append(deltaLon * m_per_deg_lon )
    
    return np.array(xaxis), np.array(yaxis) 

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def DepthNorm(name, par, depth):
    keys = np.load(data_path + name + '_keys.npy')
    means = np.load(data_path + name+ '_means.npy')
    stds = np.load(data_path + name + '_stds.npy')
    
    feature_norm = []
    for i in range(len(par)):
        depth_val = -depth[i]
        if not np.isnan(depth_val):
            indices = take_closest(keys, depth_val)
            #print(indices)
        else:
            indices = take_closest(keys, -1)
        #print(len(indices))
        #print(indices)
        #assert len(indices) >= 1
                                                    
        mean_val = means[indices]
        std_val = stds[indices]
        new_val = (par[i] - mean_val) / (std_val + 1e-3)
        feature_norm.append(new_val)
        
    
    return feature_norm

def findLargestIndex(time, specific_index):
    index = specific_index - 1
    while True:
        if index < 0 or time[index] < time[specific_index]:
            break
        else:
            index = index - 1
    return index

def NormalizeDataMultiDim(data):
    for i in range(data.shape[1]):
        if not np.isnan ((np.max(data[:, i]) - np.min(data[:,i]))):
            data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.max(data[:, i]) - np.min(data[:,i]))
    return data

def shuffle_data(data):
    shuffler = np.random.permutation(len(label))
    data_shuffled = data[shuffler]
    return data_shuffled

def data_augumentation(train_data, train_label,train_mask,train_down_mat,aug_times = 10):
    new_train_data = []
    new_train_label = []
    new_train_mask = []
    new_train_down_mat = []
    
    for i in range(len(train_label)):
        if(train_label[i]>=5):
            new_train_data.extend([train_data[i] for j in range(aug_times)])
            new_train_label.extend([train_label[i] for j in range(aug_times)])
            new_train_mask.extend([train_mask[i] for j in range(aug_times)])
            new_train_down_mat.extend([train_down_mat[i] for j in range(aug_times)])
    
    new_train_data = np.array(new_train_data)
    new_train_label = np.array(new_train_label)
    #new_train_mask = np.array(new_train_mask)
    new_train_down_mat = np.array(new_train_down_mat)
    
    train_data = np.concatenate((train_data,new_train_data), axis = 0)
    train_label = np.concatenate((train_label, new_train_label), axis = 0)
    train_mask = train_mask + new_train_mask #np.concatenate((train_mask,new_train_mask), axis = 0)
    train_down_mat = np.concatenate((train_down_mat,new_train_down_mat), axis = 0)
    
    return train_data, train_label,train_mask,train_down_mat

def evaluate(transformer, evl_input_seq, evl_output_seq, batch_val_downsample,
                              evl_input_time, evl_input_space,evl_output_time, 
                              evl_output_space,evl_src_mask,evl_trg_mask,evl_label,
             BATCH_SIZE,lr_criterion):

    num_batch = math.ceil(evl_input_seq.shape[0] / BATCH_SIZE)

    #eval_data, eval_label = shuffle_data_label(eval_data, eval_label)

    lr_pred_prob = np.zeros(evl_input_seq.shape[0])
    total_loss = 0
    prediction = []
    pred_uq = [] 
    weight_list = np.zeros((len(evl_input_seq), 1, subpoint_num ))
    
    with torch.no_grad():
        transformer.eval()
       
        for k in range(num_batch):
            s_idx = k * BATCH_SIZE

            e_idx = min(evl_input_seq.shape[0], s_idx + BATCH_SIZE)
            #feat_cut = torch.clone(eval_data[s_idx:e_idx])

            label_cut = evl_label[s_idx:e_idx]

            
            
            #evl_downsample_sub = evl_downsample[s_idx:e_idx].float().to(device)
            # for mat in evl_downsample[s_idx:e_idx]:
            #     evl_downsample_sub.append(sparse.csr_matrix.todense(mat))
            # evl_downsample_sub = torch.from_numpy(np.array(evl_downsample_sub)).float().to(device)

            lr_prob, weights, uncertainty = transformer(evl_input_seq[s_idx:e_idx], evl_output_seq[s_idx:e_idx], batch_val_downsample[k],
                              evl_input_time[s_idx:e_idx], evl_input_space[s_idx:e_idx],evl_output_time[s_idx:e_idx], 
                              evl_output_space[s_idx:e_idx],evl_src_mask[s_idx:e_idx],evl_trg_mask[s_idx:e_idx])
            weight_list[s_idx:e_idx,:] = weights.cpu().numpy()
            lr_loss = lr_criterion(lr_prob.squeeze(), 
                                   label_cut)

            total_loss += lr_loss
            if k ==0:
                prediction=lr_prob.cpu().detach().numpy()
                pred_uq = uncertainty.cpu().detach().numpy()
            else:
                prediction=np.concatenate((prediction, lr_prob.cpu().detach().numpy()), axis=0)
                pred_uq = np.concatenate((pred_uq, uncertainty.cpu().detach().numpy()), axis = 0)
                
    precision, recall, f1_score = eval_metrics(torch.from_numpy(prediction).squeeze().to(device), evl_label)
    validation_loss =  total_loss.cpu().detach().numpy() / evl_input_seq.shape[0]
    return validation_loss, np.array(prediction), np.array(pred_uq), weight_list, f1_score

def prepareDataset(features_st,label, TIME_WINDOW): 
    time = features_st[:, -1]
    features_concat = []
    label_concat = []
    for i in range(features_st.shape[0]):
        idx_end = findLargestIndex(time, i)
        idx_start = idx_end + 1  - TIME_WINDOW
        if idx_start > 0:
            features_neigh = features_st[idx_start:idx_end + 1, :]
            features_i = features_st[i,:]
            features_i = np.expand_dims(features_i, axis = 0)
            #features_i = np.expand_dims(features_i, axis = 0)
            #features_neigh = np.expand_dims(features_neigh, axis = 0)
            features_concat_i = np.concatenate((features_i, features_neigh), axis = 0)
            features_concat.append(features_concat_i)
            label_concat.append(label[i])
    return np.array(features_concat),np.array(label_concat)

home_path = '/home/whe2/STGNN/experiment/'


###downsample matrix
down_mat = []
for i in range(features_data.shape[0]):
    downsample_location = sparse_homepath + 'sparse_downsample/' + str(i) + 'downsample_mat.npz'
    down_mat.append(sparse.load_npz(downsample_location))
    


### sparse mask 


path = sparse_homepath +  'list_mask/'
if not os.path.exists(path):
    os.mkdir(path)
loc = sparse_homepath +  'list_mask/'  + 'mask_mat.pickle'
with open(loc, 'rb') as handle:
    mask_mat = pickle.load(handle)


### dataset split 


train_instance = int(np.shape(features_data)[0]*0.7)
val_instance = int(np.shape(features_data)[0]*0.8)

train_features_data = features_data[:train_instance]
train_label = label_data[:train_instance]

train_mask = mask_mat[:train_instance]
train_downsample = down_mat[:train_instance]

val_features_data = features_data[train_instance:val_instance]
val_label = label_data[train_instance:val_instance]
val_mask = mask_mat[train_instance:val_instance]
val_downsample = down_mat[train_instance:val_instance]

test_features_data = features_data[val_instance:]
test_label = label_data[val_instance:]
test_mask = mask_mat[val_instance:]
test_downsample = down_mat[val_instance:]

train_features_data, train_label,train_mask,train_downsample = data_augumentation(train_features_data, 
                                                                                train_label,train_mask,train_downsample, aug_times = 10)








shape = train_features_data.shape
train_new_features_data = torch.from_numpy(train_features_data).to(dtype=torch.float32, device= device)

shape = train_features_data.shape
train_input_seq = torch.from_numpy(train_features_data[:,:,:8]).to(dtype=torch.float32, device= device)
train_output_seq = torch.from_numpy(train_features_data[:,0,:8][:,np.newaxis,:]).to(dtype=torch.float32, device= device)

train_input_time = torch.from_numpy(train_features_data[:,:,-1]).to(dtype=torch.float32, device= device)

train_input_space = torch.from_numpy(train_features_data[:,:,-3:-1]).to(dtype=torch.float32, device= device)
train_output_time = torch.from_numpy(train_features_data[:,0,-1][:,np.newaxis]).to(dtype=torch.float32, device= device)
train_output_space = torch.from_numpy(train_features_data[:,0,-3:-1][:,np.newaxis,:]).to(dtype=torch.float32, device= device)

train_label = torch.from_numpy(train_label).to(dtype=torch.float32, device= device)

shape = val_features_data.shape
val_input_seq = torch.from_numpy(val_features_data[:,:,:8]).to(dtype=torch.float32, device= device)
val_output_seq = torch.from_numpy(val_features_data[:,0,:8][:,np.newaxis,:]).to(dtype=torch.float32, device= device)

val_input_time = torch.from_numpy(val_features_data[:,:,-1]).to(dtype=torch.float32, device= device)

val_input_space = torch.from_numpy(val_features_data[:,:,-3:-1]).to(dtype=torch.float32, device= device)
val_output_time = torch.from_numpy(val_features_data[:,0,-1][:,np.newaxis]).to(dtype=torch.float32, device= device)
val_output_space = torch.from_numpy(val_features_data[:,0,-3:-1][:,np.newaxis,:]).to(dtype=torch.float32, device= device)

val_label = torch.from_numpy(val_label).to(dtype=torch.float32, device= device)

shape = test_features_data.shape
test_input_seq = torch.from_numpy(test_features_data[:,:,:8]).to(dtype=torch.float32, device= device)
test_output_seq = torch.from_numpy(test_features_data[:,0,:8][:,np.newaxis,:]).to(dtype=torch.float32, device= device)

test_input_time = torch.from_numpy(test_features_data[:,:,-1]).to(dtype=torch.float32, device= device)

test_input_space = torch.from_numpy(test_features_data[:,:,-3:-1]).to(dtype=torch.float32, device= device)
test_output_time = torch.from_numpy(test_features_data[:,0,-1][:,np.newaxis]).to(dtype=torch.float32, device= device)
test_output_space = torch.from_numpy(test_features_data[:,0,-3:-1][:,np.newaxis,:]).to(dtype=torch.float32, device= device)

test_label = torch.from_numpy(test_label).to(dtype=torch.float32, device= device)



val_input_seq[:,0,-1] = 0
val_output_seq[:,0,-1] = 0
test_input_seq[:,0,-1] = 0
test_output_seq[:,0,-1] = 0



transformer = Transformer(
    src_vocab_size = opt['src_vocab_size'],
    trg_vocab_size = opt['trg_vocab_size'],
    src_pad_idx=opt['src_pad_idx'],
    trg_pad_idx=opt['trg_pad_idx'],
    embed_size = opt['embed_size'],
    num_layers=opt['num_layers'],
    forward_expansion=opt['forward_expansion'],
    heads=opt['heads'],
    dropout=opt['dropout'],
    device=opt['device'],
    max_length = opt['max_length']).to(opt['device'])


if regression:
    lr_criterion = torch.nn.MSELoss()
else:
    lr_criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)

lr_optimizer = torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=LEARNING_RATE, weight_decay=weight_decay)#torch.optim.SGD(transformer.parameters(),lr=LEARNING_RATE,momentum=0.9, weight_decay=5e-4) #torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(lr_optimizer, step_size=4, gamma=0.9)


#device = torch.device('cuda:{}'.format(0))
#train_downsample = train_down_mat #torch.from_numpy(np.random.rand(32, 100,100)).float().to(device)
train_src_mask = torch.IntTensor(train_mask).to(device) #torch.from_numpy(np.random.rand(32, 100, 100)).float().to(device)

train_trg_mask = torch.IntTensor([[train_mask[i][0]] for i in range(len(train_mask))]).to(device) #torch.from_numpy(train_mask[:,0,:][:,np.newaxis,:]).to(device)#torch.from_numpy(np.random.rand(32, 1, 100)).float().to(device)

#val_downsample = val_down_mat #torch.from_numpy(np.random.rand(32, 100,100)).float().to(device)
val_src_mask = torch.IntTensor(val_mask).to(device) #torch.from_numpy(val_mask).to(device) #torch.from_numpy(np.random.rand(32, 100, 100)).float().to(device)
val_trg_mask = torch.IntTensor([[val_mask[i][0]] for i in range(len(val_mask))]).to(device)#train_mask[:,0].to(device) #torch.from_numpy(val_mask[:,0,:][:,np.newaxis,:]).to(device)#torch.from_numpy(np.random.rand(32, 1, 100)).float().to(device)

#test_downsample = test_down_mat #torch.from_numpy(np.random.rand(32, 100,100)).float().to(device)
test_src_mask = torch.IntTensor(test_mask).to(device) #torch.from_numpy(np.random.rand(32, 100, 100)).float().to(device)
test_trg_mask = torch.IntTensor([[test_mask[i][0]] for i in range(len(test_mask))]).to(device)#torch.from_numpy(np.random.rand(32, 1, 100)).float().to(device)




val_instance = val_input_seq.shape[0]
#BATCH_SIZE = 512
val_num_batch = math.ceil(val_input_seq.shape[0] / BATCH_SIZE)

batch_val_downsample = []

for k in range(val_num_batch):
    s_idx = k * BATCH_SIZE

    e_idx = min(val_instance, s_idx + BATCH_SIZE)

    new_row = []
    new_col = []
    new_data = []
    i  = 0
    for val_downsample_sub in val_downsample[s_idx:e_idx]:
        val_downsample_sub = val_downsample_sub.tocoo()
        new_row = np.concatenate((new_row,longest_window*i + val_downsample_sub.row), axis = 0)
        new_col = np.concatenate((new_col,(TIME_WINDOW+1)*i + val_downsample_sub.col), axis = 0)
        new_data = np.concatenate((new_data, val_downsample_sub.data), axis=0)
        i = i + 1
    indices = torch.cat((torch.tensor(new_row[np.newaxis,:], dtype = torch.int), torch.tensor(new_col[np.newaxis,:],  dtype = torch.int)), axis = 0)
    values = torch.tensor(new_data)
    batch_val_downsample.append(torch.sparse_coo_tensor(indices, values, [longest_window*(i), (TIME_WINDOW+1)*(i)],
                                                        dtype = torch.float32, device=device))

train_instance = train_input_seq.shape[0]
num_batch = math.ceil(train_input_seq.shape[0] / BATCH_SIZE)

batch_train_downsample = []

for k in range(num_batch):
    s_idx = k * BATCH_SIZE

    e_idx = min(train_instance, s_idx + BATCH_SIZE)

    new_row = []
    new_col = []
    new_data = []
    i  = 0
    for train_downsample_sub in train_downsample[s_idx:e_idx]:
        train_downsample_sub = train_downsample_sub.tocoo()
        new_row = np.concatenate((new_row,longest_window*i + train_downsample_sub.row), axis = 0)
        new_col = np.concatenate((new_col,(TIME_WINDOW+1)*i + train_downsample_sub.col), axis = 0)
        new_data = np.concatenate((new_data, train_downsample_sub.data), axis=0)
        i = i + 1
    indices = torch.cat((torch.tensor(new_row[np.newaxis,:], dtype = torch.int), torch.tensor(new_col[np.newaxis,:],  dtype = torch.int)), axis = 0)
    values = torch.tensor(new_data)
    batch_train_downsample.append(torch.sparse_coo_tensor(indices, values, [longest_window*(i), (TIME_WINDOW+1)*(i)],
                                                          dtype = torch.float32, device=device))
            


        
def generateTrainDataset(train_input_seq,train_input_time,train_input_space, train_new_features_data,train_src_mask,batch_train_downsample,
                        num_batch,BATCH_SIZE,train_instance):
    train_batch_sample = []
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE

        e_idx = min(train_instance, s_idx + BATCH_SIZE)
        #feat_cut = torch.clone(train_data[s_idx:e_idx])
        #feat_cut[:,0, features.shape[1]-1] = 0
        for loop in [0] + list(np.random.randint(low=0, high = TIME_WINDOW, size =(number_selection-1,))):
            
            label_cut = torch.clone(train_input_seq[s_idx:e_idx, loop, -1])
            train_input_cut = torch.clone(train_input_seq[s_idx:e_idx])
            train_input_cut[:, loop, -1] = 0
            
            train_output_cut =  torch.clone(train_input_cut[:, loop,:]).unsqueeze(1)
            
            
            #train_output_cut[:, 0, -1] = 0
            train_output_time_cut = train_new_features_data[s_idx:e_idx,loop, -1].unsqueeze(1)
            train_output_space_cut = train_new_features_data[s_idx:e_idx,loop, -3:-1].unsqueeze(1)
            train_trg_mask_cut = torch.IntTensor([[train_mask[i][loop]] for i in range(s_idx, e_idx)]).to(device)
            
            train_batch_sample.append([train_input_cut, train_output_cut, batch_train_downsample[k],
                                  train_input_time[s_idx:e_idx], train_input_space[s_idx:e_idx],train_output_time_cut, 
                                  train_output_space_cut,train_src_mask[s_idx:e_idx],train_trg_mask_cut, label_cut])
    return train_batch_sample



train_upsample_data = generateTrainDataset(train_input_seq,train_input_time,train_input_space, train_new_features_data,train_src_mask,batch_train_downsample, num_batch,BATCH_SIZE,train_instance)      

print("finished train upsample data")
        



# train_upsample_data = generateTrainDataset(train_input_seq,train_input_time,train_input_space, train_new_features_data,train_src_mask,batch_train_downsample, num_batch,BATCH_SIZE,train_instance)      

# print("finished train upsample data")

        
        
training_loss = []
validation_loss = []

space_pe_weight = []
time_pe_weight = []
parm={}
eval_range = 5
n_epoch = 500
pre_valid_loss = 100

print("start training")

for epoch in (range(n_epoch)):
    transformer.train()
    #train_data, train_label = shuffle_data_label(train_data, train_label)
    
    random.shuffle(train_upsample_data)
    lr_pred_prob = np.zeros(train_input_seq.shape[0])
    lr_pred_uq = np.zeros(train_input_seq.shape[0])
    
    total_loss = 0
    
    time_start = perf_counter()
    print("time_start",time_start)
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE

        e_idx = min(train_instance, s_idx + BATCH_SIZE)
        #feat_cut = torch.clone(train_data[s_idx:e_idx])
        #feat_cut[:,0, features.shape[1]-1] = 0
        for loop in range(number_selection):

            
#             label_cut = torch.clone(train_input_seq[s_idx:e_idx, loop, -1])
#             train_input_cut = torch.clone(train_input_seq[s_idx:e_idx])
#             train_input_cut[:, loop, -1] = 0
            
#             train_output_cut =  torch.clone(train_input_cut[:, loop,:]).unsqueeze(1)
            
            
#             #train_output_cut[:, 0, -1] = 0
#             train_output_time_cut = train_new_features_data[s_idx:e_idx,loop, -1].unsqueeze(1)
#             train_output_space_cut = train_new_features_data[s_idx:e_idx,loop, -3:-1].unsqueeze(1)
#             train_trg_mask_cut = train_src_mask[s_idx:e_idx, loop,:].unsqueeze(1)
            
        
            lr_optimizer.zero_grad()
            


    #         train_downsample_sub = []
    #         for mat in train_downsample[s_idx:e_idx]:
    #             train_downsample_sub.append(sparse.csr_matrix.todense(mat))
    #         train_downsample_sub = torch.from_numpy(np.array(train_downsample_sub)).float().to(device)



            lr_prob, weights,uncertainty = transformer(train_upsample_data[int(k*number_selection)+loop][0], train_upsample_data[int(k*number_selection)+loop][1], 
                                           train_upsample_data[int(k*number_selection)+loop][2],
                                  train_upsample_data[int(k*number_selection)+loop][3], train_upsample_data[int(k*number_selection)+loop][4],
                                           train_upsample_data[int(k*number_selection)+loop][5], 
                                  train_upsample_data[int(k*number_selection)+loop][6],train_upsample_data[int(k*number_selection)+loop][7],train_upsample_data[int(k*number_selection)+loop][8])
        
            #lr_loss =(torch.div( torch.square(lr_prob.squeeze() - train_upsample_data[int(k*number_selection)+loop][9]), (prior_uq - uncertainty)) + torch.log(torch.max(prior_uq - uncertainty, epison))).mean()
            
            lr_loss =  lr_criterion(lr_prob.squeeze(), 
                                    train_upsample_data[int(k*number_selection)+loop][9])

            lr_loss.backward()
            lr_optimizer.step()
            total_loss += lr_loss.cpu().detach().numpy()
    time_stop = perf_counter()
    print("elapse time: ", time_stop - time_start)
    scheduler.step(epoch = epoch)
    print('Epoch {}, lr {}'.format(epoch, lr_optimizer.param_groups[0]['lr']))
    for name, param in transformer.named_parameters():
        #print(name)
        #print("name: ", name, "param: ", param) # space_encoder.Wr.weight time_encoder.basis_freq
        if 'space_encoder.Wr.weight' in name:
            parm[name]=param.cpu().detach().numpy()  
            space_pe_weight.append(parm['space_encoder.Wr.weight'])
        if 'time_encoder.basis_freq' in name:
            #print(param)
            parm[name]=param.cpu().detach().numpy()
            time_pe_weight.append(parm['time_encoder.basis_freq'])        
    #scheduler.step()
            #torch.save(tgan.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))

    print("epoch: ", epoch, "training loss: ", total_loss / train_instance)
    if epoch%eval_range == 0:
        evaluation_loss,prediction,pred_uq, weight, f1_score = evaluate(transformer, val_input_seq, val_output_seq, batch_val_downsample,
                              val_input_time, val_input_space,val_output_time, 
                              val_output_space,val_src_mask,val_trg_mask,val_label,
             BATCH_SIZE,lr_criterion)
        #print("pred_prob: ", pred_prob)
        print("epoch: ", epoch, "validation loss: ",evaluation_loss, "validation f1 score", f1_score.cpu().detach().numpy())
        training_loss.append(lr_loss)
        validation_loss.append(evaluation_loss)
        
        
        
        if evaluation_loss < pre_valid_loss: 
            path = val_save_folder + "tgan_space_iter_" + str(epoch) +"_" +str(np.round(f1_score.cpu().detach().numpy(),2))+ ".pt"
            torch.save(transformer.state_dict(), path)
        pre_valid_loss = evaluation_loss
path = val_save_folder +  "tgan_space_iter_" + str(epoch) +"_" +str(np.round(f1_score.cpu().detach().numpy(),2))+ ".pt"
torch.save(transformer.state_dict(), path)   

