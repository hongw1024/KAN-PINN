"""
Benchmark 1: Advection-Dispersion Equation (ADE) - Instantaneous Release
=========================================================================
Solves the 1D advection-dispersion equation with an instantaneous point
source (Gaussian pulse) using both KAN-PINN and MLP-PINN frameworks.

Governing equation:
    dc/dt + v * dc/dx - D * d2c/dx2 + k * c = 0

Analytical solution:
    C(x, t) = mp / (2 * sqrt(pi * D * t)) * exp(-(x - v*t)^2 / (4*D*t))

Authors: Hongwei Lei, Gaoyuan Li, Jie Niu
Repository: https://github.com/hongw1024/KAN-PINN
License: MIT
"""


# ======================================================================
import numpy as np
import torch
import torch.nn as nn
import os
import scipy.optimize
from scipy.interpolate import griddata
from SALib.sample import sobol_sequence
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.autograd import grad
from kan import *
import time
from datetime import datetime
import matplotlib as mpl
from utils.plotting import colormap_parula
# Set Times New Roman font
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'    # Use STIX math font, compatible with Times New Roman


dtype = torch.float64  
torch.set_default_dtype(dtype)   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1108)
np.random.seed(1108)

# ======================================================================
# Parameter configuration
vel = 1.0
diff = 1e-3
decay = 0.0
mp = 1
Lx = 1.0  # space domain size
Tf = 1.0  # time domain size


def gen_analytical(Lx=1., Tf=1., xn=201, tn=101, v=1.0, D= 1e-3, mp = 1 ):

    X, T = np.meshgrid(np.linspace(0, Lx, xn), np.linspace(0.04, Tf,  tn))

    C = mp / (2 * np.sqrt(np.pi * D * T)) * np.exp(-(X - v * T) ** 2 / (4 * D * T))

    C[0, 0] = 0     # remove the initial C

    return X[0, :], T[:, 0], C


# Compute analytical solution

x, t, Exact = gen_analytical(Lx=Lx, Tf=Tf, xn=201, tn=101, v=vel, D=diff, mp = mp)

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))


Exact = Exact.reshape(X.shape)


# Generate training data
N_u = 200    # Number of initial/boundary points
N_f = 10000   # Number of collocation points

lb = X_star.min(axis=0)    # Lower bounds [x_min, t_min]
ub = X_star.max(axis=0)    # Upper bounds [x_max, t_max]

# Initial and boundary conditions
ic = np.hstack((X[0:1, :].T, T[0:1, :].T))
uic = Exact[0:1, :].T

bcl = np.hstack((X[:, 0:1], T[:, 0:1]))
ulb = Exact[:, 0:1]

bcu = np.hstack((X[:, -1:], T[:, -1:]))
uub = Exact[:, -1:]

X_u_train = np.vstack([ic, bcl, bcu])
u_train = np.vstack([uic, ulb, uub])

# Generate collocation points
#X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = lb + (ub - lb) * sobol_sequence.sample(N_f, 2)

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]

X_f_train = np.vstack((X_f_train, X_u_train))

# Plotting
fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(211)
h = ax.imshow(Exact, interpolation='nearest', cmap=colormap_parula,
                extent=[x.min(), x.max(), t.min(), t.max()],
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(h, cax=cax)
ax.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', label='U data (%d points)' % (X_u_train.shape[0]),
        markersize=4, clip_on=False)
ax.plot(X_f_train[:, 0], X_f_train[:, 1], 'r.', label='F data (%d points)' % (X_f_train.shape[0]),
        markersize=4, clip_on=False)
ax.legend(frameon=False, loc='best')
ax.set_title('$Q(x, t)$', fontsize=14)
ax.set_ylabel('$t\ (d)$')

# ======================================================================
class ADE_MLP(nn.Module):
    def __init__(self, X_u, u, X_f, layers, lb, ub, vel, diff, decay,X_star, Exact):
        super(ADE_MLP, self).__init__()
        
# Normalization bounds
        self.lb = torch.tensor(lb, dtype=dtype).to(device)
        self.ub = torch.tensor(ub, dtype=dtype).to(device)
        
# Training data
        self.x_u = torch.tensor(X_u[:,0:1], dtype=dtype, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:,1:2], dtype=dtype, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=dtype).to(device)
        
# Collocation points
        self.x_f = torch.tensor(X_f[:,0:1], dtype=dtype, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:,1:2], dtype=dtype, requires_grad=True).to(device)
        
        self.vel = vel
        self.diff = diff
        self.decay = decay
        self.layers = layers
        self.activation = nn.Tanh()
# Full-domain reference data
        self.X_star = torch.tensor(X_star, dtype=dtype).to(device)
        self.Exact_flat = torch.tensor(Exact.flatten()[:, None], dtype=dtype).to(device)
# Neural network architecture
        self.model = self.build_model()
        
# Training records
        self.losses = []
        self.u_errors = []
        self.f_errors = []
        self.L2_errors = []
        self.log_frequency = 10
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.iter = 0
    
    def build_model(self):
        layers = []
        for i in range(len(self.layers)-1):
            layers.append(nn.Linear(self.layers[i], self.layers[i+1]))
            if i < len(self.layers)-2:    # No activation on the last layer
                layers.append(nn.Tanh())
        return nn.Sequential(*layers).to(device)
    
    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
# Normalize to [-1, 1] range
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(X)
    
    def net_u(self, x, t):
        return self(x, t)
    
    def net_f(self, x, t):
        u = self.net_u(x, t)
        
# Retain computation graph for derivative computation
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        return u_t + self.vel * u_x - self.diff * u_xx + self.decay * u
    
    def loss_fn(self):
        
        u_pred_train = self.net_u(self.x_u, self.t_u)
        loss_u = torch.mean((self.u - u_pred_train)**2)
        
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_f = torch.mean(f_pred**2)
        
        return loss_u + loss_f, loss_u, loss_f
    
    def L2_error(self):
               
        with torch.no_grad():
            u_pred_full = self.net_u(self.X_star[:, 0:1], self.X_star[:, 1:2])
            L2_error = (torch.linalg.norm(u_pred_full - self.Exact_flat) / 
                       torch.linalg.norm(self.Exact_flat))
        return L2_error
    
    def train_model(self, Adam_iters, LBFGS_iters=None):
# Adam optimization phase
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print("--- Starting Adam Optimization ---")
        for it in range(Adam_iters):
            self.iter += 1
            optimizer.zero_grad()
            total_loss, loss_u, loss_f = self.loss_fn()
            L2_errors = self.L2_error()

            total_loss.backward()
            optimizer.step()

            self.losses.append(total_loss.item())
            self.u_errors.append(loss_u.item())
            self.f_errors.append(loss_f.item())
            self.L2_errors.append(L2_errors.item())

            if it % self.log_frequency == 0:
                elapsed = self.get_elapsed()
                epoch_duration = self.get_epoch_duration()

                print(f'Adam Epoch: {it:6d}  Loss: {total_loss.item():.8e}  '
                      f'U Loss: {loss_u.item():.8e}  F Loss: {loss_f.item():.8e}  '
                      f'L2_errors: {L2_errors.item():.8e}  '
                      f'Elapsed: {elapsed}(+{epoch_duration})')
        
# L-BFGS optimization phase
        if LBFGS_iters is not None:
            print("--- Starting L-BFGS Optimization ---")
            optimizer = torch.optim.LBFGS(self.parameters(),max_iter=LBFGS_iters, max_eval=LBFGS_iters, history_size=50, tolerance_grad=1e-20, 
                                          tolerance_change=1e-20, line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                total_loss, loss_u, loss_f = self.loss_fn()
                L2_errors = self.L2_error()
                total_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.losses.append(total_loss.item())
                self.u_errors.append(loss_u.item())
                self.f_errors.append(loss_f.item())
                self.L2_errors.append(L2_errors.item())
                if self.iter % self.log_frequency == 0:
                    elapsed = self.get_elapsed()
                    
                    epoch_duration = self.get_epoch_duration()

                    
                    print(f'LBFGS Epoch: {self.iter:6d}  Loss: {total_loss.item():.8e}  '
                        f'U Loss: {loss_u.item():.8e}  F Loss: {loss_f.item():.8e}  '
                        f'L2_errors: {L2_errors.item():.8e}  '
                        f'Elapsed: {elapsed}(+{epoch_duration})')
                self.iter += 1
                return total_loss
            optimizer.step(closure)
        print("==================")
        print(f"Training finished (Total epochs {Adam_iters + self.iter}): duration = {self.get_elapsed()}")

        plt.figure()
        plt.plot(self.losses, label="Losses")
        plt.plot(self.u_errors, label="$Errors_{loss_u}$")
        plt.plot(self.f_errors, label="$Errors_{loss_f}$")
        plt.yscale("log")
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Errors", fontsize=14)
        plt.legend(fontsize=14)
        return self.losses, self.u_errors, self.f_errors,self.L2_errors

    
    def get_epoch_duration(self):
        now = time.time()
        duration = datetime.fromtimestamp(now - self.prev_time).strftime("%S.%f")[:-5]
        self.prev_time = now
        return duration
    
    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")
    
    def predict(self, X_star):
# Enable gradient computation for prediction
        x = torch.tensor(X_star[:, 0:1], dtype=dtype, requires_grad=True).to(device)
        t = torch.tensor(X_star[:, 1:2], dtype=dtype, requires_grad=True).to(device)
        
        with torch.no_grad():
            c_pred = self.net_u(x, t)
        
        return c_pred.cpu().numpy()

# ======================================================================
class ADE_KAN(nn.Module):
    def __init__(self, X_u, u, X_f,  lb, ub, vel, diff, decay, X_star, Exact):
        super(ADE_KAN, self).__init__()
        
# Normalization bounds
        self.lb = torch.tensor(lb, dtype=dtype).to(device)
        self.ub = torch.tensor(ub, dtype=dtype).to(device)
        
# Training data
        self.x_u = torch.tensor(X_u[:,0:1], dtype=dtype, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:,1:2], dtype=dtype, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=dtype).to(device)
        
# Collocation points
        self.x_f = torch.tensor(X_f[:,0:1], dtype=dtype, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:,1:2], dtype=dtype, requires_grad=True).to(device)
        
        self.vel = vel
        self.diff = diff
        self.decay = decay
# Full-domain reference data
        self.X_star = torch.tensor(X_star, dtype=dtype).to(device)
        self.Exact_flat = torch.tensor(Exact.flatten()[:, None], dtype=dtype).to(device)
        
# Neural network architecture
        self.KAN =KAN(width=[2, 8, 8, 1], grid=20, k=5, base_fun=torch.nn.Tanh(), device=device).to(torch.float64)
        
# Training records
        self.losses = []
        self.u_errors = []
        self.f_errors = []
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.log_frequency = 10
        self.iter = 0
        self.L2_errors = []
        #self.Exact = torch.tensor(Exact, dtype=torch.float64).to(device)
    
    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
# Normalize to [-1, 1] range
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        
        return self.KAN(X)
    
    def net_u(self, x, t):
        return self(x, t)
    
    def net_f(self, x, t):
        u = self.net_u(x, t)
        
# Retain computation graph for derivative computation
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        return u_t + self.vel * u_x - self.diff * u_xx + self.decay * u
    
    def loss_fn(self):
# Data loss at training points
        u_pred_train = self.net_u(self.x_u, self.t_u)
        loss_u = torch.mean((self.u - u_pred_train)**2)
        
# Physics constraint
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_f = torch.mean(f_pred**2)
        
        return loss_u + loss_f, loss_u, loss_f
    
    def L2_error(self):
# Full-domain L2 error (no gradient tracking)
        with torch.no_grad():
            u_pred_full = self.net_u(self.X_star[:, 0:1], self.X_star[:, 1:2])
            L2_error = (torch.linalg.norm(u_pred_full - self.Exact_flat) / 
                       torch.linalg.norm(self.Exact_flat))
        return L2_error
    
    def train_model(self, Adam_iters, LBFGS_iters=None):
# Adam optimization phase
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print("--- Starting Adam Optimization ---")
        for it in range(Adam_iters):
            self.iter += 1
            optimizer.zero_grad()
            total_loss, loss_u, loss_f = self.loss_fn()
            L2_errors = self.L2_error()
            total_loss.backward()
            optimizer.step()
            self.losses.append(total_loss.item())
            self.u_errors.append(loss_u.item())
            self.f_errors.append(loss_f.item())
            self.L2_errors.append(L2_errors.item())

            if it % self.log_frequency == 0:
                elapsed = self.get_elapsed()
                epoch_duration = self.get_epoch_duration()

                print(f'Adam Epoch: {it:6d}  Loss: {total_loss.item():.8e}  '
                      f'U Loss: {loss_u.item():.8e}  F Loss: {loss_f.item():.8e}  '
                      f'L2_errors: {L2_errors.item():.8e}  '
                      f'Elapsed: {elapsed}(+{epoch_duration})')
        
# L-BFGS optimization phase
        if LBFGS_iters is not None:
            print("--- Starting L-BFGS Optimization ---")
            optimizer = torch.optim.LBFGS(self.parameters(),max_iter=LBFGS_iters, max_eval=LBFGS_iters, history_size=50, tolerance_grad=1e-20, 
                                          tolerance_change=1e-20, line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                total_loss, loss_u, loss_f = self.loss_fn()
                L2_errors = self.L2_error()
                total_loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.losses.append(total_loss.item())
                self.u_errors.append(loss_u.item())
                self.f_errors.append(loss_f.item())
                self.L2_errors.append(L2_errors.item())
                if self.iter % self.log_frequency == 0:
                    elapsed = self.get_elapsed()
                    
                    epoch_duration = self.get_epoch_duration()

                    
                    print(f'LBFGS Epoch: {self.iter:6d}  Loss: {total_loss.item():.8e}  '
                        f'U Loss: {loss_u.item():.8e}  F Loss: {loss_f.item():.8e}  '
                        f'L2_errors: {L2_errors.item():.8e}  '
                        f'Elapsed: {elapsed}(+{epoch_duration})')
                self.iter += 1
                return total_loss
            optimizer.step(closure)
        print("==================")
        print(f"Training finished (Total epochs {Adam_iters + self.iter}): duration = {self.get_elapsed()}")

        plt.figure()
        plt.plot(self.losses, label="Losses")
        plt.plot(self.u_errors, label="$Errors_{loss_u}$")
        plt.plot(self.f_errors, label="$Errors_{loss_f}$")
        plt.yscale("log")
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Errors", fontsize=14)
        plt.legend(fontsize=14)
        return self.losses, self.u_errors, self.f_errors,self.L2_errors

    
    def get_epoch_duration(self):
        now = time.time()
        duration = datetime.fromtimestamp(now - self.prev_time).strftime("%S.%f")[:-5]
        self.prev_time = now
        return duration
    
    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")
    
    def predict(self, X_star):
# Enable gradient computation for prediction
        x = torch.tensor(X_star[:, 0:1], dtype=dtype, requires_grad=True).to(device)
        t = torch.tensor(X_star[:, 1:2], dtype=dtype, requires_grad=True).to(device)
        
# Use no_grad for u_pred; f_pred requires gradients
        with torch.no_grad():
            c_pred = self.net_u(x, t)
        
        return c_pred.cpu().numpy()

# ======================================================================
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

mlp_model = ADE_MLP(X_u_train, u_train, X_f_train, layers, lb, ub, vel, diff, decay, X_star, Exact)
kan_model =ADE_KAN(X_u_train, u_train, X_f_train, lb, ub, vel, diff, decay, X_star, Exact)
X_star_tensor = torch.tensor(X_star, dtype=dtype, device=device)

# ======================================================================
# Parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nMLP Parameters: {count_parameters(mlp_model):,}")
print(f"KAN Parameters: {count_parameters(kan_model):,}")

# ======================================================================
mlp_losses, mlp_u_errors, mlp_f_errors,mlp_L2_errors = mlp_model.train_model(Adam_iters=10001, LBFGS_iters=0)

# ======================================================================
kan_losses, kan_u_errors, kan_f_errors, kan_L2_errors = kan_model.train_model(Adam_iters=10001, LBFGS_iters=0)

# ======================================================================
kan_pred  = kan_model.predict(X_star)
mlp_pred = mlp_model.predict(X_star)
kan_pred = kan_pred.reshape(X.shape)
mlp_pred=mlp_pred.reshape(X.shape)

print(np.mean(mlp_pred-Exact)**2)
print(np.mean(kan_pred-Exact)**2)

# ======================================================================
mlp_error_c = np.abs(mlp_pred - Exact)
kan_error_c = np.abs(kan_pred - Exact)
print(f"mlp_error_c Relative L2 Error: {np.linalg.norm(mlp_error_c)/np.linalg.norm(Exact):.4e}")
print(f"kan_error_c Relative L2 Error: {np.linalg.norm(kan_error_c)/np.linalg.norm(Exact):.4e}")

# ======================================================================
output_dir = "CASE1 results"
os.makedirs(output_dir, exist_ok=True)

# Create comparison figure: ground truth and loss curves (side by side)
fig = plt.figure(figsize=(18, 6), dpi=150)    # Wide rectangular layout
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.4)    # 1-row 2-column layout

# ============== Font size parameters ===============
title_font = 18           # Title font size
axis_label_font = 18      # Axis label font size
tick_font = 18            # Tick font size
legend_font = 18          # Legend font size
cbar_font = 18            # Colorbar font size
# ========================================

# Ground truth concentration field (left panel)
ax0 = fig.add_subplot(gs[0])
h0 = ax0.imshow(Exact, cmap=colormap_parula,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
ax0.set_title('(a) Truth Concentration Field', fontsize=title_font)
ax0.set_ylabel('Temporal Domain ($t$)', fontsize=axis_label_font)
ax0.set_xlabel('Spatial Domain ($x$)', fontsize=axis_label_font)
ax0.tick_params(axis='both', labelsize=tick_font)    # Increase tick font size

divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(h0, cax=cax, label='Concentration')
cbar.set_label('Concentration', rotation=270, labelpad=25, fontsize=cbar_font)
cbar.ax.tick_params(labelsize=tick_font)

# Loss curve comparison (right panel)
ax1 = fig.add_subplot(gs[1])
ax1.semilogy(mlp_losses, label='MLP Total Loss', linewidth=1)
ax1.semilogy(kan_losses, label='KAN Total Loss', linewidth=1)
ax1.set_title('(b) Training Loss Curves', fontsize=title_font)
ax1.set_xlabel('Iterations', fontsize=axis_label_font)
ax1.set_ylabel('Loss Value ', fontsize=axis_label_font)
ax1.legend(fontsize=legend_font, loc='upper right', frameon=False)
ax1.set_facecolor('white')
ax1.tick_params(axis='both', labelsize=tick_font)

plt.savefig(f"{output_dir}/case1_groundtruth_losses.png", 
           bbox_inches='tight', 
           dpi=600,
           facecolor='white')
plt.show()

# ======================================================================
# Part 2: Prediction visualization (side by side)
fig = plt.figure(figsize=(18, 6), dpi=150)    # Horizontal layout

# 1-row 4-column grid: KAN | KAN colorbar | MLP | MLP colorbar
gs = gridspec.GridSpec(1, 2, 
                       width_ratios=[1, 1],    # Equal width for main panels, narrow colorbars
                       wspace=0.4, hspace=0.1)            # Adjust horizontal spacing


# (c) KAN prediction
ax2 = fig.add_subplot(gs[0, 0])
h2 = ax2.imshow(kan_pred, cmap=colormap_parula,
                extent=[x.min(), x.max(), t.min(), t.max()],
                origin='lower', aspect='auto',
                vmin=Exact.min(), vmax=Exact.max())
ax2.set_title('(c) KAN Prediction', fontsize=title_font)
ax2.set_ylabel('Temporal Domain ($t$)', fontsize=axis_label_font)
ax2.set_xlabel('Spatial Domain ($x$)', fontsize=axis_label_font)
ax2.tick_params(axis='both', labelsize=tick_font)
# Use standard marker style to avoid label issues
ax2.scatter(X_u_train[:, 0], X_u_train[:, 1], 
            s=5, c='black', marker='x', alpha=0.5)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = fig.colorbar(h2, cax=cax2)
cbar2.set_label('Concentration', rotation=270, labelpad=25, fontsize=cbar_font)
cbar2.ax.tick_params(labelsize=tick_font)

# (d) MLP prediction
ax3 = fig.add_subplot(gs[0, 1])
h3 = ax3.imshow(mlp_pred, cmap=colormap_parula,
                extent=[x.min(), x.max(), t.min(), t.max()],
                origin='lower', aspect='auto',
                vmin=Exact.min(), vmax=Exact.max())
ax3.set_title('(d) MLP Prediction', fontsize=title_font)
ax3.set_ylabel('Temporal Domain ($t$)', fontsize=axis_label_font)
ax3.set_xlabel('Spatial Domain ($x$)', fontsize=axis_label_font)
ax3.tick_params(axis='both', labelsize=tick_font)
# Use standard marker style to avoid label issues
ax3.scatter(X_u_train[:, 0], X_u_train[:, 1], 
            s=5, c='black', marker='x', alpha=0.5)
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.1)
cbar3 = fig.colorbar(h3, cax=cax3)
cbar3.set_label('Concentration', rotation=270, labelpad=25, fontsize=cbar_font)
cbar3.ax.tick_params(labelsize=tick_font)

# Save figure
plt.savefig(f"{output_dir}/case1_prediction_comparison_horizontal.png", 
            bbox_inches='tight', 
            dpi=600,
            facecolor='white')
plt.show()

# ======================================================================
# Part 1: Error heatmap comparison (side by side)
kan_log_error = np.log10(kan_error_c + 1e-10)
mlp_log_error = np.log10(mlp_error_c + 1e-10)

# Create professional layout canvas - horizontal layout
fig = plt.figure(figsize=(18, 6), dpi=150)    # Increased width for side-by-side layout

# 1-row grid: KAN | MLP | colorbar
gs = gridspec.GridSpec(1, 2,  
                       width_ratios=[1, 1],    # Equal width for main panels, narrow colorbars
                       wspace=0.4)      # Adjust horizontal spacing

# ============== Font size parameters ===============
title_font = 18           # Title font size
axis_label_font = 18      # Axis label font size
tick_font = 18            # Tick font size
legend_font = 18          # Legend font size
cbar_font = 18            # Colorbar font size
# ========================================

# (e) KAN logarithmic error heatmap
ax4 = fig.add_subplot(gs[0, 0])
h4 = ax4.imshow(kan_log_error, cmap=colormap_parula,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto', 
               vmin=-4, vmax=0)
ax4.set_title('(e) KAN Prediction Error', fontsize=title_font)
ax4.set_ylabel('Temporal Domain ($t$)', fontsize=axis_label_font)
ax4.set_xlabel('Spatial Domain ($x$)', fontsize=axis_label_font)
ax4.tick_params(axis='both', labelsize=tick_font)
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("right", size="5%", pad=0.1)
cbar4 = fig.colorbar(h4, cax=cax4)
cbar4.set_label('log10(|Error|)', rotation=270, labelpad=25, fontsize=cbar_font)
cbar4.ax.tick_params(labelsize=tick_font)

# (f) MLP logarithmic error heatmap
ax5 = fig.add_subplot(gs[0, 1])
h5 = ax5.imshow(mlp_log_error, cmap=colormap_parula,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto',
               vmin=-4, vmax=0)
ax5.set_title('(f) MLP Prediction Error', fontsize=title_font)
ax5.set_ylabel('Temporal Domain ($t$)', fontsize=axis_label_font)
ax5.set_xlabel('Spatial Domain ($x$)', fontsize=axis_label_font)
ax5.tick_params(axis='both', labelsize=tick_font)
divider = make_axes_locatable(ax5)
cax5 = divider.append_axes("right", size="5%", pad=0.1)
cbar5 = fig.colorbar(h5, cax=cax5)
cbar5.set_label('log10(|Error|)', rotation=270, labelpad=25, fontsize=cbar_font)
cbar5.ax.tick_params(labelsize=tick_font)

# Professional save settings
plt.savefig(f"{output_dir}/case1 error_heatmaps_horizontal.png", 
            bbox_inches='tight', 
            dpi=600,
            facecolor='white',
            transparent=False,
            metadata={'Description': 'Prediction error comparison between KAN and MLP models',
                      'Author': 'Your Name'})
plt.show()

# ======================================================================
# Create performance comparison report
report = f"""
Case4 Performance Report (Variable Concentration Source)
-------------------------------------------------------
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Device: {'GPU' if torch.cuda.is_available() else 'CPU'}

Model Parameters:
  MLP: {count_parameters(mlp_model):,} parameters
  KAN: {count_parameters(kan_model):,} parameters

MIN Relative L2 Errors:
  MLP: {np.min(mlp_L2_errors):.4e}
  KAN: {np.min(kan_L2_errors):.4e}


MIN TOTAL  LOSS:
  mlp: {np.min(mlp_losses):.4e}
  kan: {np.min(kan_losses):.4e}
  
Maximum Absolute Errors:
  MLP: {np.max(mlp_error_c):.4e}
  KAN: {np.max(kan_error_c):.4e}

Training Details:
  Training points: {N_u}
  Collocation points: {N_f}
  Adam iterations: 10001
"""

with open(f"{output_dir}/case4_performance_report.txt", "w") as f:
    f.write(report)

print(report)

# ======================================================================
# Create professional layout canvas - 3 rows x 2 columns
fig = plt.figure(figsize=(18, 14.4), dpi=150)
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1],    # Adjust height ratios
                       hspace=0.36, wspace=0.3)

# Font size parameters
title_font = 24
axis_label_font = 24
tick_font = 24
cbar_font = 24

# Define colormaps
pred_cmap = colormap_parula    # Colormap for prediction plots
error_cmap = colormap_parula    # Colormap for error plots

# (a) KAN prediction
ax0 = fig.add_subplot(gs[0, 0])
h0 = ax0.imshow(kan_pred, cmap=pred_cmap,
                extent=[x.min(), x.max(), t.min(), t.max()],
                origin='lower', aspect='auto',
                vmin=kan_pred.min(), vmax=kan_pred.max())
ax0.set_title('(a) KAN Prediction', fontsize=title_font)
ax0.set_ylabel(r'$t\ (T)$', fontsize=axis_label_font)    # LaTeX format
ax0.scatter(X_u_train[:, 0], X_u_train[:, 1], 
            s=5, c='white', marker='x', alpha=0.7)
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar0 = fig.colorbar(h0, cax=cax0)
#cbar0.set_label('Concentration', rotation=270, labelpad=25, fontsize=cbar_font)
cbar0.ax.tick_params(labelsize=tick_font)

# (b) KAN logarithmic error heatmap
ax1 = fig.add_subplot(gs[0, 1])
h1 = ax1.imshow(kan_log_error, cmap=error_cmap,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto', 
               vmin=-4, vmax=0)
ax1.set_title(r'(b) $\log_{10}|C_{KAN} - C_{ana}|$', fontsize=title_font)
ax1.set_ylabel(r'$t\ (T)$', fontsize=axis_label_font)    # LaTeX format
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = fig.colorbar(h1, cax=cax1)
# Format as log10|Error| with subscript
#cbar1.set_label(r'$\log_{10}|\mathrm{Error}|$', rotation=270, labelpad=25, fontsize=cbar_font)
cbar1.ax.tick_params(labelsize=tick_font)

# (c) MLP prediction
ax2 = fig.add_subplot(gs[1, 0])
h2 = ax2.imshow(mlp_pred, cmap=pred_cmap,
                extent=[x.min(), x.max(), t.min(), t.max()],
                origin='lower', aspect='auto',
                vmin=mlp_pred.min(), vmax=mlp_pred.max())
ax2.set_title('(c) MLP Prediction', fontsize=title_font)
ax2.set_ylabel(r'$t\ (T)$', fontsize=axis_label_font)    # LaTeX format
ax2.scatter(X_u_train[:, 0], X_u_train[:, 1], 
            s=5, c='white', marker='x', alpha=0.7)
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = fig.colorbar(h2, cax=cax2)
#cbar2.set_label('Concentration', rotation=270, labelpad=25, fontsize=cbar_font)
cbar2.ax.tick_params(labelsize=tick_font)

# (d) MLP logarithmic error heatmap
ax3 = fig.add_subplot(gs[1, 1])
h3 = ax3.imshow(mlp_log_error, cmap=error_cmap,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto',
               vmin=-4, vmax=0)
ax3.set_title(r'(d) $\log_{10}|C_{MLP} - C_{ana}|$', fontsize=title_font)
ax3.set_xlabel(r'$x\ (L)$', fontsize=axis_label_font)    # LaTeX format
ax3.set_ylabel(r'$t\ (T)$', fontsize=axis_label_font)    # LaTeX format
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.1)
cbar3 = fig.colorbar(h3, cax=cax3)
# Format as log10|Error| with subscript
#cbar3.set_label(r'$\log_{10}|\mathrm{Error}|$', rotation=270, labelpad=25, fontsize=cbar_font)
cbar3.ax.tick_params(labelsize=tick_font)

# (e) Analytical solution
ax4 = fig.add_subplot(gs[2, 0])
h4 = ax4.imshow(Exact, cmap=pred_cmap,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
ax4.set_title('(e) Analytical solution', fontsize=title_font)
ax4.set_xlabel(r'$x\ (L)$', fontsize=axis_label_font)    # LaTeX format
ax4.set_ylabel(r'$t\ (T)$', fontsize=axis_label_font)    # LaTeX format
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("right", size="5%", pad=0.1)
cbar4 = fig.colorbar(h4, cax=cax4)
#cbar4.set_label('Concentration', rotation=270, labelpad=25, fontsize=cbar_font)
cbar4.ax.tick_params(labelsize=tick_font)

# (f) Training loss curves
ax5 = fig.add_subplot(gs[2, 1])
ax5.semilogy(mlp_losses, label='MLP', linewidth=0.5, color='blue')
ax5.semilogy(kan_losses, label='KAN', linewidth=0.5, color='red')
ax5.set_title('(f) Training Loss Curves', fontsize=title_font)
ax5.set_xlabel('Iterations', fontsize=axis_label_font)
ax5.set_ylabel('Loss Value', fontsize=axis_label_font)
ax5.legend(fontsize=axis_label_font, loc='upper right', frameon=False)
ax5.set_facecolor('white')
ax5.grid(False)
ax5.tick_params(axis='both', labelsize=tick_font)

# Uniformly set tick font size for all subplots
for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
    ax.tick_params(axis='both', labelsize=tick_font)

# Save high-quality figure
plt.savefig(f"{output_dir}/case1_3x2_visualization.png", 
           bbox_inches='tight', 
           dpi=600, 
           facecolor='white')

plt.show()
