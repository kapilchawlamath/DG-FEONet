import torch
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
from pprint import pprint


# ARGS
parser = argparse.ArgumentParser("SEM")
## Data
parser.add_argument("--equation", type=str, default='Standard', choices=['Standard', 'varcoeff', 'burgers','bdrylayer', 'inputcoeff'])
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--num_data", type=int, default=3000)
parser.add_argument("--file", type=str, help='Example: --file 3000N18')
parser.add_argument("--basis_order", type=int, help='P1->d=1, P2->d=2')
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])
parser.add_argument("--bdry", type=str, default='dirichlet', choices=['dirichlet', 'neumann','bdry_layer'])

args = parser.parse_args()
gparams = args.__dict__

EQUATION = gparams['equation']
EPS = gparams['eps']

NUM_DATA = gparams['num_data']
FILE = gparams['file']
BASIS_ORDER = gparams['basis_order']
KIND = gparams['kind']
BDRY = gparams['bdry']

# Seed
if KIND=='train':
    np.random.seed(5)
elif KIND=='validate':
    np.random.seed(10)
else:
    print('error!')

mesh=np.load('mesh_1DP{}/ne{}_{}.npz'.format(BASIS_ORDER,int(FILE.split('N')[1].split('_')[0]), EQUATION))
NUM_ELEMENT, NUM_PTS, p, c = mesh['ne'], mesh['ng'], mesh['p'], mesh['c']
NUM_BASIS = NUM_PTS

#STIFF=mesh['stiff']
STIFF = mesh[f'{KIND}_stiff']
CONV=mesh['convection']
#LOAD_VECTOR=mesh['load_vector']
LOAD_VECTOR = mesh[f'{KIND}_load_vector']  

if NUM_DATA!=int(FILE.split('N')[0]) or NUM_ELEMENT!=int(FILE.split('N')[1].split('_')[0]):
    print("Error!! : Please check --file with --num_data and --N")

def c_input(x, coeff):
    c_left, c_mid, c_right, x0, x1 = coeff
    c_val = np.where(x < x0, c_left,
              np.where(x < x1, c_mid, c_right))
    return c_val, coeff


def standard(n,eps, stiff, conv, load_vector, kind):
    input_matrix=mesh[f'{kind}_matrix'][n]
    f_vector = mesh[f'{kind}_load_vector'][n]
    S = stiff[n]
    C=conv
    coeff_u=np.linalg.solve(S+conv+input_matrix, f_vector)
    return coeff_u

def create_data(num_data, p, eps, stiff, conv, load_vector, kind):
    data = []
    for n in tqdm(range(num_data)):
        c_value, coeff_c=c_input(p,mesh[f'{kind}_coeff_inputs'][n])
        coeff_u= standard(n, eps, stiff, conv, load_vector, kind)
        data.append([coeff_u, c_value, coeff_c])
    return np.array(data, dtype=object)

def save_obj(data, name, kind, basis_order):
    cwd = os.getcwd()
    path = os.path.join(cwd,'data', f'P{basis_order}', kind)
    if os.path.isdir(path) == False:
        os.makedirs(path)
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
         
data = create_data(NUM_DATA, p, EPS, STIFF, CONV, LOAD_VECTOR, KIND)
save_obj(data, FILE, KIND, BASIS_ORDER)