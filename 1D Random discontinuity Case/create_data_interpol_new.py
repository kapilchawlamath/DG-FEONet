import torch
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
import scipy
from pprint import pprint
from dolfin import *
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np

# ARGS
parser = argparse.ArgumentParser("SEM")
## Data
parser.add_argument("--equation", type=str, default='Standard', choices=['Standard', 'varcoeff', 'burgers', 'inputcoeff'])
parser.add_argument("--num_data", type=int, default=3000)
parser.add_argument("--file", type=str, help='Example: --file 3000N18')
parser.add_argument("--basis_order", type=int, help='P1->d=1, P2->d=2')
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])
parser.add_argument("--bdry", type=str, default='dirichlet', choices=['dirichlet', 'neumann','bdry_layer'])
parser.add_argument("--exact", type=int, default=300)


args = parser.parse_args()
gparams = args.__dict__

EQUATION = gparams['equation']

NUM_DATA = gparams['num_data']
FILE = gparams['file']
BASIS_ORDER = gparams['basis_order']
KIND = gparams['kind']
BDRY = gparams['bdry']
NUM_ELEMENT_exact = gparams['exact']


# Seed
if KIND=='train':
    np.random.seed(5)
elif KIND=='validate':
    np.random.seed(10)
else:
    print('error!')

# Load exact data for interpolation
mesh=np.load('mesh_1DP{}/ne{}_{}.npz'.format(BASIS_ORDER,NUM_ELEMENT_exact,EQUATION))
if NUM_ELEMENT_exact==mesh['ne']:
    p_exact = mesh['p']
    pickle_file = f'1000N'+str(NUM_ELEMENT_exact)+'_'+EQUATION
    with open(f'data/P{BASIS_ORDER}/{KIND}/' + pickle_file + '.pkl', 'rb') as f:
        data_exact = pickle.load(f)

mesh_data=np.load('mesh_1DP{}/ne{}_{}.npz'.format(BASIS_ORDER,int(FILE.split('N')[1].split('_')[0]), EQUATION))
NUM_ELEMENT, NUM_PTS, p, c = mesh['ne'], mesh['ng'], mesh['p'], mesh['c']
NUM_BASIS = NUM_PTS

if NUM_DATA!=int(FILE.split('N')[0]) or NUM_ELEMENT!=int(FILE.split('N')[1].split('_')[0]):
    print("Error!! : Please check --file with --num_data and --N")
def c_input(x, coeff):
    c_left, c_mid, c_right, x0, x1 = coeff
    c_val = np.where(x < x0, c_left,
              np.where(x < x1, c_mid, c_right))
    return c_val, coeff



def create_data(num_data, mesh, p_exact, data_exact, kind, coeff_inputs, BASIS_ORDER):
    data = []
    V_dg = FunctionSpace(mesh, "DG", BASIS_ORDER)

    for n in tqdm(range(num_data)):
        u_highres = data_exact[n][0]
        c_value, coeff_c = c_input(p_exact, coeff_inputs[n])

        # Step 1: Rebuild fine function
        V_fine = FunctionSpace(mesh, "DG", BASIS_ORDER)
        u_fine = Function(V_fine)
        u_fine.vector().set_local(u_highres)

        # Step 2: L² projection
        u = TrialFunction(V_dg)
        v = TestFunction(V_dg)
        a_proj = inner(u, v) * dx
        L_proj = inner(u_fine, v) * dx
        u_proj = Function(V_dg)
        solve(a_proj == L_proj, u_proj)

        coeff_u = u_proj.vector().get_local()
        data.append([coeff_u, c_value, coeff_c])

    return np.array(data, dtype=object)


def save_obj(data, name, kind, basis_order):
    cwd = os.getcwd()
    path = os.path.join(cwd,'data', f'P{basis_order}', kind)
    if os.path.isdir(path) == False:
        os.makedirs(path)
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

from dolfin import *

# Reconstruct mesh for FEniCS from scratch
num_elements = int(FILE.split('N')[1].split('_')[0])
coeff_inputs = mesh_data[f'{KIND}_coeff_inputs']
mesh_fenics = IntervalMesh(num_elements, -1, 1)

data = create_data(
    num_data=NUM_DATA,
    mesh=mesh_fenics,
    p_exact=p_exact,
    data_exact=data_exact,
    kind=KIND,  
    coeff_inputs=coeff_inputs,
    BASIS_ORDER=BASIS_ORDER
)



#data = create_data(NUM_DATA, p, KIND)      
save_obj(data, FILE, KIND, BASIS_ORDER)

# Choose a sample index from the newly created data array
plot_idx = np.random.randint(data.shape[0])
print(f"Validation Sample Index: {plot_idx}")

# Extract values directly from your L²-projected dataset
coeff_u, c_value, coeff_c_vals = data[plot_idx]

# Print the coefficient vector used to define c(x)
print("\nCoefficient vector of c(x): [m0, m1, n0, n1] =", coeff_c_vals)

# Print interpolated u(x) and c(x) at each DOF point
print("\n{:<12s} {:<12s} {:<12s}".format("x", "c(x)", "u(x)"))
print("-" * 36)
for i in range(len(c_value)):
    print("{:<12.5f} {:<12.5f} {:<12.5f}".format(p[i], c_value[i], coeff_u[i]))


