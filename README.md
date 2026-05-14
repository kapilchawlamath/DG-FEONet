# Discontinuous Galerkin Finite Element Operator Network for Solving Non-smooth PDEs

This is a FEniCS and PyTorch implementation of the Discontinuous Galerkin Finite Element Operator Network (DG-FEONet) for solving parametric PDEs with discontinuous coefficients and non-smooth solutions.

DG-FEONet predicts the discontinuous Galerkin finite element coefficients using a neural network. The network is trained in a data-free manner by minimizing the residual of the DG weak formulation.

## 1. 1D_random_discontinuity: randomly located discontinuities in 1D

In the folder `"1D_random_discontinuity"`:

This example solves the one-dimensional convection-diffusion-reaction problem

$$
-\frac{d}{dx}\left(\varepsilon(x)\frac{du}{dx}\right)
+ b\frac{du}{dx}
+ c(x)u
=
f(x),
\qquad x\in (-1,1),
$$

with homogeneous Dirichlet boundary conditions

$$
u(-1)=u(1)=0.
$$

The convection coefficient is fixed as

$$
b=0.01.
$$

The input is the discontinuous reaction coefficient \(c(x)\), defined by

$$
c(x)=
\begin{cases}
c_0, & x < x_0,\\
c_1, & x_0 \le x < x_1,\\
c_2, & x \ge x_1,
\end{cases}
$$

where

$$
c_0\in [0,5),\qquad c_1\in [5,10),\qquad c_2\in [10,15].
$$

The source term and diffusion coefficient are also piecewise constant on the same subintervals:

$$
f(x)=
\begin{cases}
1.0, & x < x_0,\\
-1.5, & x_0 \le x < x_1,\\
2.5, & x \ge x_1,
\end{cases}
$$

and

$$
\varepsilon(x)=
\begin{cases}
0.01, & x < x_0,\\
0.02, & x_0 \le x < x_1,\\
0.03, & x \ge x_1.
\end{cases}
$$

The discontinuity locations \(x_0\) and \(x_1\) are randomly selected from the mesh element boundaries.

### Step1 - Generate and save the exact DG data

To generate the training data, use

```bash
python3 create_data_exact_new.py --equation inputcoeff --num_data 1000 --file 1000N32_inputcoeff --basis_order 1 --kind train
```

To generate the validation data, use

```bash
python3 create_data_exact_new.py --equation inputcoeff --num_data 1000 --file 1000N32_inputcoeff --basis_order 1 --kind validate
```

### Step2 - Save the train and validation data by interpolation

Using the `create_data_interpol_new.py` code, generate the interpolated training data by

```bash
python3 create_data_interpol_new.py --equation inputcoeff --file 1000N32_inputcoeff --num_data 1000 --basis_order 1 --kind train --exact 32
```

Generate the interpolated validation data by

```bash
python3 create_data_interpol_new.py --equation inputcoeff --file 1000N32_inputcoeff --num_data 1000 --basis_order 1 --kind validate --exact 32
```

Here, `--exact 32` means that the exact DG data is generated on a mesh with 32 elements.

### Step3 - Train DG-FEONet

To train the DG-FEONet model, use

```bash
python3 -u FEONet_1D_new.py test --seed 0 --gpu 0 --equation inputcoeff --file 1000N32_inputcoeff --basis_order 1 --bdry dirichlet --model FCNN --blocks 4 --ks 5 --filters 32 --epochs 50000 | tee data/P1/train/1000ne32_inputcoeff.out
```


### Step4 - Plot the results

After training, run the plotting script:

```bash
python3 plot_results.py
```

## Notes

The current 1D example uses homogeneous Dirichlet boundary conditions. Therefore, the boundary terms in the linear functional involving the prescribed boundary value vanish. The SIPG bilinear form still includes the boundary consistency and penalty terms needed to weakly impose the boundary condition.

Large data files, trained model checkpoints, and raw training outputs are not included in this repository. They can be regenerated using the commands above.