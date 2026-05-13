import random
import torch
import time
import datetime
import subprocess
import os
import argparse
import gc
import sys
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import diags
from pprint import pprint


import matplotlib.pyplot as plt
from net.network import *


# ARGS
parser = argparse.ArgumentParser("SEM")
parser.add_argument('name', type=str, help='experiments name')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_squeue", action='store_true')
parser.add_argument("--gpu", type=int, default=0)

## Data
parser.add_argument("--equation", type=str, default='Standard', choices=['Standard', 'varcoeff', 'burgers','bdrylayer', 'inputcoeff'])
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--b", type=float, default=-1)
parser.add_argument("--file", type=str, default='3000N32', help='Example: --file 3000N18_dirichlet')
parser.add_argument("--basis_order", type=int, default=1, help='P1->d=1, P2->d=2')
parser.add_argument("--bdry", type=str, default='dirichlet', choices=['dirichlet', 'neumann'])

## Train parameters
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--model", type=str, default='Net2D', choices=['FCNN','CNN'])
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32, choices=[8, 16, 32, 64])
parser.add_argument("--act", type=str, default='silu')
parser.add_argument("--loss", type=str, default='MSE', choices=['MAE', 'MSE', 'RMSE', 'RelMSE'])
parser.add_argument("--epochs", type=int, default=80000)
parser.add_argument("--pre_epochs", type=int, default=0)

args = parser.parse_args()
gparams = args.__dict__

NAME=gparams['name']

## Random seed
random_seed=gparams['seed']
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

import torch


## Choose gpu
if not gparams['use_squeue']:
    use_cuda = torch.cuda.is_available()
    print("Is CUDA available? :", use_cuda)
    
    if use_cuda:
        gpu_id = str(gparams['gpu'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print("-> Using GPU number", gpu_id)
    else:
        print("-> No GPU detected, using CPU.")


#Equation
EQUATION = gparams['equation']
EPS = gparams['eps']
FILE = gparams['file']
NUM_DATA = int(FILE.split('N')[0])
BASIS_ORDER = gparams['basis_order']
BDRY = gparams['bdry']

mesh=np.load('mesh_1DP{}/ne{}_{}.npz'.format(BASIS_ORDER,int(FILE.split('N')[1].split('_')[0]),EQUATION))
NUM_ELEMENT, NUM_PTS, p, c = mesh['ne'], mesh['ng'], mesh['p'], mesh['c']
NUM_BASIS = NUM_PTS

#STIFF=mesh['stiff']
CONV=mesh['convection']
TRAIN_STIFF = mesh['train_stiff']      # shape (NUM_TRAIN, N, N)
VAL_STIFF   = mesh['validate_stiff']   # shape (NUM_VAL, N, N)




#Model
models = {
          'FCNN': FCNN,
          'CNN': CNN,
          }
MODEL = models[gparams['model']]
BLOCKS = int(gparams['blocks'])
KERNEL_SIZE = int(gparams['ks'])
FILTERS = int(gparams['filters'])
PADDING = (KERNEL_SIZE - 1)//2
ACT = gparams['act']

# Select the correct model from the dictionary
D_in = 1
D_out = NUM_BASIS
model = MODEL(ACT, D_in, FILTERS, D_out, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)

# Move model to the correct device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#Train
EPOCHS = int(gparams['epochs'])
D_in = 1
D_out = NUM_BASIS
if gparams['batch_size']==None: #Full-batch
    BATCH_SIZE = NUM_DATA
else:
    BATCH_SIZE = gparams['batch_size']


    
#Save file
cur_time = str(datetime.datetime.now()).replace(' ', 'T')
cur_time = cur_time.replace(':','').split('.')[0].replace('-','')
FOLDER = f'{gparams["model"]}_epochs{EPOCHS}_{cur_time}'
PATH = os.path.join('train', f'P{BASIS_ORDER}', FILE, FOLDER)





# CREATE PATHING
if os.path.isdir(PATH) == False: os.makedirs(PATH); os.makedirs(os.path.join(PATH, 'pics'))
elif os.path.isdir(PATH) == True:
    if args.pretrained is None:
        print("\n\nPATH ALREADY EXISTS!\n\nEXITING\n\n")
        exit()
    else:
        print("\n\nPATH ALREADY EXISTS!\n\nLOADING MODEL\n\n")

class Dataset(Dataset):
    def __init__(self, gparams, mesh, kind='train'):
        self.pickle_file = gparams['file']
        with open(f'data/P{BASIS_ORDER}/{kind}/' + self.pickle_file + '.pkl', 'rb') as f:
            self.data = pickle.load(f)

        self.input_matrix = mesh[f'{kind}_matrix']      # (num_data, N, N)
        self.load_vector  = mesh[f'{kind}_load_vector'] # (num_data, N)
        self.stiff_matrix = mesh[f'{kind}_stiff']       # (num_data, N, N)  <-- NEW

    def __getitem__(self, idx):
        coeff_u    = torch.FloatTensor(self.data[idx,0]).unsqueeze(0)
        c_value    = torch.FloatTensor(self.data[idx,1]).unsqueeze(0)
        coeff_c    = torch.FloatTensor(self.data[idx,2])
        input_mat  = torch.FloatTensor(self.input_matrix[idx])
        load_vec   = torch.FloatTensor(self.load_vector[idx])
        stiff_mat  = torch.FloatTensor(self.stiff_matrix[idx])  # <-- NEW

        return {
            'coeff_u': coeff_u,
            'c_value': c_value,
            'coeff_c': coeff_c,
            'input_matrix': input_mat,
            'load_vector': load_vec,
            'stiff': stiff_mat,        # <-- NEW
        }

    def __len__(self):
        return len(self.data)

lg_dataset = Dataset(gparams, mesh, kind='train')
trainloader = DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=True)
lg_dataset = Dataset(gparams, mesh, kind='validate')
validateloader = DataLoader(lg_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Num train : {}, Num test: {}".format(len(trainloader.dataset), len(validateloader.dataset)))



model = MODEL(ACT, D_in, FILTERS, D_out, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)

# SEND TO GPU (or CPU)
if torch.cuda.is_available():
    model.cuda()
else:
    model.to("cpu")

    
# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

model.apply(weights_init)


def init_optim(model):
    params = {'history_size': 10,
              'tolerance_grad': 1E-15,
              'tolerance_change': 1E-15,
              'max_eval': 10,
                }
    return torch.optim.LBFGS(model.parameters(), **params)

optimizer = init_optim(model)

loss_func = torch.nn.MSELoss(reduction="sum")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#STIFF = torch.tensor(STIFF, device=device, dtype=torch.float32)
CONV = torch.tensor(CONV, device=device, dtype=torch.float32)
#LOAD_VECTOR = torch.tensor(LOAD_VECTOR, device=device, dtype=torch.float32)

def weak_form(coeff_u, input_matrix, stiff_batch, conv, load_vec_batch):
    # coeff_u: [B, 1, N] -> [B, N, N]
    coeff_u = coeff_u.repeat(1, coeff_u.shape[-1], 1)

    # conv is global [N,N] → broadcast to [B,N,N]
    A = stiff_batch + conv + input_matrix.squeeze(1)  # all [B, N, N]

    LHS = A * coeff_u
    LHS = torch.sum(LHS, dim=-1)   # [B, N]

    RHS = load_vec_batch.to(device)
    return LHS, RHS

def closure(model, c_value, input_matrix, stiff_batch, conv, load_vec_batch):
    pred_coeff_u = model(c_value)  # [B, 1, N]

    LHS, RHS = weak_form(pred_coeff_u, input_matrix, stiff_batch, conv, load_vec_batch)

    loss_wf = torch.zeros((NUM_BASIS,), device=device)
    for ii in range(NUM_BASIS):
        loss_wf[ii] = loss_func(LHS[:, ii], RHS[:, ii])

    loss = torch.sum(loss_wf)
    return loss, pred_coeff_u

def rel_L2_error(pred, true):
    return (torch.sum((true-pred)**2, dim=-1)/torch.sum((true)**2, dim=-1))**0.5

def log_gparams(gparams):
    cwd = os.getcwd()
    os.chdir(PATH)
    with open('parameters.txt', 'w') as f:
        for k, v in gparams.items():
            if k == 'losses':
                df = pd.DataFrame(gparams['losses'])
                df.to_csv('losses.csv')
            else:
                entry = f"{k}:{v}\n"
                f.write(entry)
    os.chdir(cwd)


def log_path(path):
    with open(os.path.expanduser("~/paths.txt"), "a") as f:  # Saves in your home directory
    #with open("../../paths.txt", "a") as f:
        f.write(str(path) + '\n')
        f.close()
log_path(PATH)
log_gparams(gparams)
################################################
time0 = time.time()
losses=[]
train_rel_L2_errors=[]
test_rel_L2_errors=[]
for epoch in range(1, EPOCHS+1):
    model.train()
    loss_total = 0
    num_samples=0
    train_rel_L2_error = 0

    for batch_idx, sample_batch in enumerate(trainloader):
        optimizer.zero_grad()
        coeff_u = sample_batch['coeff_u']
        c_value = sample_batch['c_value'].to(device)
        input_matrix = sample_batch['input_matrix'].to(device)
        load_vector = sample_batch['load_vector'].to(device)  

        stiff_batch = sample_batch['stiff'].to(device)
        loss,u_pred = closure(model, c_value, input_matrix, stiff_batch, CONV, load_vector)


        loss.backward()  

        optimizer.step(loss.item)
        loss_total += np.round(float(loss.item()), 4)
        num_samples += coeff_u.shape[0]

        with torch.no_grad():
            model.eval()
            _,u_pred = closure(model, c_value, input_matrix, stiff_batch, CONV, load_vector)
            u_pred=u_pred.squeeze().detach().cpu()
            coeff_u=coeff_u.squeeze()
            train_rel_L2_error += torch.sum(rel_L2_error(u_pred, coeff_u))

    train_rel_L2_error /= num_samples
    

    if epoch%100==0:
        ## Test
        num_samples=0
        test_rel_L2_error = 0
        for batch_idx, sample_batch in enumerate(validateloader):
            with torch.no_grad():
                model.eval()
                coeff_u = sample_batch['coeff_u']
                c_value = sample_batch['c_value'].to(device)
                input_matrix = sample_batch['input_matrix'].to(device)
                stiff_batch = sample_batch['stiff'].to(device)
                load_vector = sample_batch['load_vector'].to(device)
                loss,u_pred = closure(model, c_value, input_matrix, stiff_batch, CONV, load_vector)

                _,u_pred = closure(model, c_value, input_matrix, stiff_batch, CONV, load_vector)
                u_pred=u_pred.squeeze().detach().cpu()
                coeff_u=coeff_u.squeeze()
                test_rel_L2_error += torch.sum(rel_L2_error(u_pred, coeff_u))

                num_samples += coeff_u.shape[0]
        test_rel_L2_error /= num_samples
        
        ##Save and print
        losses.append(loss_total)
        train_rel_L2_errors.append(train_rel_L2_error)
        test_rel_L2_errors.append(test_rel_L2_error)
        torch.save({'model_state_dict': model.state_dict(),
                    'losses': losses,
                    'train_rel_L2_errors': train_rel_L2_errors,
                    'test_rel_L2_errors': test_rel_L2_errors
        }, PATH + '/model.pt')
        print("Epoch {0:4d}: weak_form_loss {1:4.1f}, train_rel_error {2:.5f}, test_rel_error {3:.5f}".format(epoch, loss_total, train_rel_L2_error, test_rel_L2_error))

        
torch.save({'model_state_dict': model.state_dict(),
            'losses': losses,
            'train_rel_L2_errors': train_rel_L2_errors,
            'test_rel_L2_errors': test_rel_L2_errors
}, PATH + '/model.pt')
        
train_t=time.time()-time0
NPARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)

gparams['train_time'] = train_t
gparams['nParams'] = NPARAMS
gparams['batchSize'] = BATCH_SIZE
gparams['path'] = PATH

log_gparams(gparams)
