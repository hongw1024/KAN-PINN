"""
Benchmark: Saint-Venant Equation (SVE) - Kinematic Wave Approximation
========================================================================
Solves the 1D kinematic wave equation (shock wave benchmark) using both
KAN-PINN and MLP-PINN frameworks.

Governing equation:
    dA/dt + dQ/dx = 0,  Q = alpha * A^(5/3)

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
#from kan import *
import time
from datetime import datetime
from fast_bspline_kan import KAN
# Plotting
from utils.plotting import colormap_parula
import matplotlib as mpl
from matplotlib import font_manager
# Set Times New Roman font
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'    # Use STIX math font, compatible with Times New Roman
# Set global font sizes
title_font = 18
axis_label_font = 18
tick_font = 18
cbar_font = 18
legend_font = 18


layers = [2, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 1]
dtype = torch.float64  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(567567)
np.random.seed(567567)

# ======================================================================
# === Numerical solution generation ===
Q0 = 2.5
S0 = 0.004
B = 15.0
n = 0.07

N_u = 2000
N_f = 8000

alpha = (n * B ** (2 / 3) / np.sqrt(S0)) ** (3 / 5)

# Spatiotemporal grid
L = 1.0
dx = 0.00125
Tf = 1.0
dt = 0.001
x = np.arange(0, L + dx, dx, dtype=np.float64)
t = np.arange(0, Tf + dt, dt, dtype=np.float64)
X, T = np.meshgrid(x, t)

Q = np.ones_like(X) * Q0
for i, ts in enumerate(T[:, 0]):
    Q[i, 0] += Q0 * np.sin(np.pi * ts)

# Finite difference solver
for l in range(len(t) - 1):
    for i in range(1, len(x)):
        Q[l + 1, i] = Q[l, i] + dt * (Q[l, i] ** (2 / 5) / (alpha * (3 / 5) * dx) * (Q[l, i - 1] - Q[l, i]))

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
lb = np.min(X_star, axis=0).astype(np.float64)
ub = np.max(X_star, axis=0).astype(np.float64)

# === Training data preparation ===
ic = np.hstack((X[0:1, :].T, T[0:1, :].T)).astype(np.float64)
Qic = Q[0:1, :].T.astype(np.float64)

bcl = np.hstack((X[:, 0:1], T[:, 0:1])).astype(np.float64)
Qlb = Q[:, 0:1].astype(np.float64)
bcu = np.hstack((X[:, -1:], T[:, -1:])).astype(np.float64)
Qub = Q[:, -1:].astype(np.float64)

X_u_train = np.vstack([ic, bcl, bcu]).astype(np.float64)
Q_u_train = np.vstack([Qic, Qlb, Qub]).astype(np.float64)

# Random sampling
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
Q_u_train = Q_u_train[idx, :]

# Generate collocation points
X_f_train = lb + (ub - lb) * sobol_sequence.sample(N_f, 2)
X_f_train = np.vstack((X_f_train, X_u_train)).astype(np.float64)

# Plotting
fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(211)

# cmap=colormap_parula,
h = ax.imshow(Q, interpolation='nearest', cmap=colormap_parula,
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


plt.show()

# ======================================================================
# === Define PINN model ===
class SVE_PINN(nn.Module):
    def __init__(self, X_u_train, u_train, X_f_train, layers, lb, ub, alpha, X_star, Q):
        super().__init__()
        
        self.x_u = torch.tensor(X_u_train[:,0], dtype=dtype, device=device).unsqueeze(1)
        self.t_u = torch.tensor(X_u_train[:,1], dtype=dtype, device=device).unsqueeze(1)
        self.u_train = torch.tensor(u_train, dtype=dtype, device=device)
        
        self.x_f = torch.tensor(X_f_train[:,0], dtype=dtype, device=device).unsqueeze(1).requires_grad_(True)
        self.t_f = torch.tensor(X_f_train[:,1], dtype=dtype, device=device).unsqueeze(1).requires_grad_(True)
        
        self.lb = torch.tensor(lb, dtype=dtype, device=device)
        self.ub = torch.tensor(ub, dtype=dtype, device=device)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=dtype, device=device))

        self.iter = 0
        self.losses = []
        self.err_Q_u = []
        self.err_Q_f = []
        self.L2_errors = []
        self.log_frequency = 100
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.X_star = torch.tensor(X_star, dtype=dtype).to(device)
        self.Exact_Q = torch.tensor(Q.flatten()[:, None], dtype=dtype).to(device)

        self.layers = layers
        self.model = self.build_model()

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
        x.requires_grad_(True)
        t.requires_grad_(True)
        Q = self.forward(x, t)
        
        Q_safe = torch.clamp(Q, min=0.1*Q0, max=5*Q0)
        
        Q_t = grad(Q_safe, t, grad_outputs=torch.ones_like(Q_safe), create_graph=True)[0]
        Q_x = grad(Q_safe, x, grad_outputs=torch.ones_like(Q_safe), create_graph=True)[0]
        
        loss_Q_f = Q_t + (Q_safe.pow(2.0/5.0) / (self.alpha * (3.0/5.0))) * Q_x
        return loss_Q_f
    
    def L2_error(self):
# Full-domain L2 error (no gradient tracking)
        with torch.no_grad():
            u_pred_full = self.net_u(self.X_star[:, 0:1], self.X_star[:, 1:2])
            L2_error = (torch.linalg.norm(u_pred_full - self.Exact_Q) / 
                       torch.linalg.norm(self.Exact_Q))
        return L2_error
    
    def loss_fn(self):        
        Q_pred = self.forward(self.x_u, self.t_u)
        loss_Q_u = torch.mean((self.u_train - Q_pred)**2)
        
        loss_Q_f = self.net_f(self.x_f, self.t_f)
        loss_Q_f = torch.mean(loss_Q_f**2)
        
        total_loss = loss_Q_u + loss_Q_f
        return total_loss, loss_Q_u, loss_Q_f
    
    def train_model(self, Adam_iters, LBFGS_iters=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print("--- Starting Adam Optimization ---")
        for it in range(Adam_iters):
            self.iter += 1
            optimizer.zero_grad()
            total_loss, loss_Q_u, loss_Q_f = self.loss_fn()
            L2_errors = self.L2_error()
            total_loss.backward()
            optimizer.step()

            self.losses.append(total_loss.item())
            self.err_Q_u.append(loss_Q_u.item())
            self.err_Q_f.append(loss_Q_f.item())
            self.L2_errors.append(L2_errors.item())

            if it % self.log_frequency == 0:
                elapsed = self.get_elapsed()
                epoch_duration = self.get_epoch_duration()
                print(f'Epoch {it:6d}: Loss={total_loss.item():.8e}, 'f'Q_u={loss_Q_u.item():.8e} Q_f={loss_Q_f.item():.8e}', f'L2_errors: {L2_errors.item():.8e},  'f'Elapsed: {elapsed}(+{epoch_duration})')
# L-BFGS optimization phase
        if LBFGS_iters is not None:
            print("--- Starting L-BFGS Optimization ---")
            optimizer = torch.optim.LBFGS(self.parameters(), max_iter=LBFGS_iters, max_eval=LBFGS_iters, history_size=100, tolerance_grad=1e-20, 
                                          tolerance_change=1e-20, line_search_fn='strong_wolfe')
        
            def closure():
                optimizer.zero_grad()
                total_loss, loss_Q_u, loss_Q_f = self.loss_fn()
                total_loss.backward()
                
                self.losses.append(total_loss.item())
                self.err_Q_u.append(loss_Q_u.item())
                self.err_Q_f.append(loss_Q_f.item())
                L2_errors = self.L2_error()
                self.L2_errors.append(L2_errors.item())
                if self.iter % 100 == 0:
                    print(f'L-BFGS Step {self.iter}: Loss={total_loss.item():.8e}, 'f'Q_u={loss_Q_u.item():.8e} Q_f={loss_Q_f.item():.8e}', f'L2_errors: {L2_errors.item():.8e},  'f'Elapsed: {elapsed}(+{epoch_duration})')
                    
                self.iter += 1
                return total_loss
            
            optimizer.step(closure)

        plt.figure()
        plt.plot(self.losses, label="Total Loss")
        plt.plot(self.err_Q_u, label="Q_u Loss")
        plt.plot(self.err_Q_f, label="Q_f Loss")
        plt.yscale("log")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        return self.losses, self.err_Q_u, self.err_Q_f, self.L2_errors
    
    def get_epoch_duration(self):
        now = time.time()
        duration = datetime.fromtimestamp(now - self.prev_time).strftime("%S.%f")[:-5]
        self.prev_time = now
        return duration
    
    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")
    
    def predict(self, X_star):
        self.eval()
# Safe tensor conversion
        if isinstance(X_star, np.ndarray):
            x = torch.as_tensor(X_star[:,0:1], dtype=dtype, device=device)
            t = torch.as_tensor(X_star[:,1:2], dtype=dtype, device=device)
        else:
            x = X_star[:,0:1].clone().detach().to(device)
            t = X_star[:,1:2].clone().detach().to(device)
        
        with torch.no_grad():
            Q_pred = self.forward(x, t)
        return Q_pred.cpu().numpy()

# ======================================================================
# === Define PINN model ===
class SVE_PIKAN(nn.Module):
    def __init__(self, X_u_train, u_train, X_f_train, lb, ub, alpha, X_star, Q):
        super().__init__()
        
        self.x_u = torch.tensor(X_u_train[:,0], dtype=dtype, device=device).unsqueeze(1)
        self.t_u = torch.tensor(X_u_train[:,1], dtype=dtype, device=device).unsqueeze(1)
        self.u_train = torch.tensor(u_train, dtype=dtype, device=device)
        
        self.x_f = torch.tensor(X_f_train[:,0], dtype=dtype, device=device).unsqueeze(1).requires_grad_(True)
        self.t_f = torch.tensor(X_f_train[:,1], dtype=dtype, device=device).unsqueeze(1).requires_grad_(True)
        
        self.lb = torch.tensor(lb, dtype=dtype, device=device)
        self.ub = torch.tensor(ub, dtype=dtype, device=device)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=dtype, device=device))
        
        self.iter = 0
        self.losses = []
        self.err_Q_u = []
        self.err_Q_f = []
        self.L2_errors = []
        self.log_frequency = 100
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.X_star = torch.tensor(X_star, dtype=dtype).to(device)
        self.Exact_Q = torch.tensor(Q.flatten()[:, None], dtype=dtype).to(device)

        self.KAN =KAN(width=[2, 10, 10, 10, 1], grid=25, k=5, base_fun=torch.nn.Tanh(),  device=device).to(torch.float64)
    
    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
# Normalize to [-1, 1] range
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.KAN(X)
    
    def net_u(self, x, t):
        return self(x, t)
    
    def net_f(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        Q = self.forward(x, t)
        
        Q_safe = torch.clamp(Q, min=0.1*Q0, max=5*Q0)
        
        Q_t = grad(Q_safe, t, grad_outputs=torch.ones_like(Q_safe), create_graph=True)[0]
        Q_x = grad(Q_safe, x, grad_outputs=torch.ones_like(Q_safe), create_graph=True)[0]
        
        loss_Q_f = Q_t + (Q_safe.pow(2.0/5.0) / (self.alpha * (3.0/5.0))) * Q_x
        
        return loss_Q_f
    
    def L2_error(self):
# Full-domain L2 error (no gradient tracking)
        with torch.no_grad():
            u_pred_full = self.net_u(self.X_star[:, 0:1], self.X_star[:, 1:2])
            L2_error = (torch.linalg.norm(u_pred_full - self.Exact_Q) / 
                       torch.linalg.norm(self.Exact_Q))
        return L2_error
    
    def loss_fn(self):        
        Q_pred = self.forward(self.x_u, self.t_u)
        loss_Q_u = torch.mean((self.u_train - Q_pred)**2)
        
        loss_Q_f = self.net_f(self.x_f, self.t_f)
        loss_Q_f = torch.mean(loss_Q_f**2)
        
        total_loss = loss_Q_u + loss_Q_f
        return total_loss, loss_Q_u, loss_Q_f
    
    def train_model(self, Adam_iters, LBFGS_iters=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print("--- Starting Adam Optimization ---")

        for it in range(Adam_iters):
            self.iter += 1
            optimizer.zero_grad()
            total_loss, loss_Q_u, loss_Q_f = self.loss_fn()
            L2_errors = self.L2_error()
            total_loss.backward()
            optimizer.step()
            self.losses.append(total_loss.item())
            self.err_Q_u.append(loss_Q_u.item())
            self.err_Q_f.append(loss_Q_f.item())
            self.L2_errors.append(L2_errors.item())

            if it % self.log_frequency == 0:
                elapsed = self.get_elapsed()
                epoch_duration = self.get_epoch_duration()
                print(f'Epoch {it:6d}: Loss={total_loss.item():.8e}, 'f'Q_u={loss_Q_u.item():.8e} Q_f={loss_Q_f.item():.8e}', f'L2_errors: {L2_errors.item():.8e},  'f'Elapsed: {elapsed}(+{epoch_duration})')
# L-BFGS optimization phase
        if LBFGS_iters is not None:
            print("--- Starting L-BFGS Optimization ---")
            optimizer = torch.optim.LBFGS(self.parameters(),max_iter=LBFGS_iters, max_eval=LBFGS_iters, history_size=100, tolerance_grad=1e-20, 
                                          tolerance_change=1e-20, line_search_fn='strong_wolfe')
        
            def closure():
                optimizer.zero_grad()
                total_loss, loss_Q_u, loss_Q_f = self.loss_fn()
                total_loss.backward()
                
                self.losses.append(total_loss.item())
                self.err_Q_u.append(loss_Q_u.item())
                self.err_Q_f.append(loss_Q_f.item())
                L2_errors = self.L2_error()
                self.L2_errors.append(L2_errors.item())
                if self.iter % 20 == 0:
                    print(f'L-BFGS Step {self.iter}: Loss={total_loss.item():.8e}, 'f'Q_u={loss_Q_u.item():.8e} Q_f={loss_Q_f.item():.8e}', f'L2_errors: {L2_errors.item():.8e},  'f'Elapsed: {elapsed}(+{epoch_duration})')
                    
                self.iter += 1
                return total_loss
            
            optimizer.step(closure)

        plt.figure()
        plt.plot(self.losses, label="Total Loss")
        plt.plot(self.err_Q_u, label="Q_u Loss")
        plt.plot(self.err_Q_f, label="Q_f Loss")
        plt.yscale("log")
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        return self.losses, self.err_Q_u, self.err_Q_f, self.L2_errors
    
    def get_epoch_duration(self):
        now = time.time()
        duration = datetime.fromtimestamp(now - self.prev_time).strftime("%S.%f")[:-5]
        self.prev_time = now
        return duration
    
    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")
    
    def predict(self, X_star):
        self.eval()
# Safe tensor conversion
        if isinstance(X_star, np.ndarray):
            x = torch.as_tensor(X_star[:,0:1], dtype=dtype, device=device)
            t = torch.as_tensor(X_star[:,1:2], dtype=dtype, device=device)
        else:
            x = X_star[:,0:1].clone().detach().to(device)
            t = X_star[:,1:2].clone().detach().to(device)
        
        with torch.no_grad():
            Q_pred = self.forward(x, t)
        return Q_pred.cpu().numpy()

# ======================================================================
mlp_model = SVE_PINN(X_u_train, Q_u_train, X_f_train, layers, lb, ub, alpha, X_star, Q).double().to(device)
kan_model = SVE_PIKAN(X_u_train, Q_u_train, X_f_train, lb, ub, alpha, X_star, Q).double().to(device)

# ======================================================================
# Parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nMLP Parameters: {count_parameters(mlp_model):,}")
print(f"KAN Parameters: {count_parameters(kan_model):,}")

# ======================================================================
mlp_losses, mlp_err_u, mlp_err_f, mlp_L2_errors = mlp_model.train_model(Adam_iters=10001, LBFGS_iters=0)

# ======================================================================
kan_losses, kan_err_u, kan_err_f, kan_L2_errors = kan_model.train_model(Adam_iters=10001, LBFGS_iters=0)

# ======================================================================
kan_pred  = kan_model.predict(X_star)
mlp_pred  = mlp_model.predict(X_star)
kan_pred = kan_pred.reshape(X.shape)
mlp_pred  = mlp_pred.reshape(X.shape)
mlp_error_Q = np.abs(mlp_pred - Q)
kan_error_Q = np.abs(kan_pred - Q)


print(f"mlp_error_Q Relative L2 Error: {np.linalg.norm(mlp_error_Q)/np.linalg.norm(Q):.4e}")
print(f"kan_error_Q Relative L2 Error: {np.linalg.norm(kan_error_Q)/np.linalg.norm(Q):.4e}")

# ======================================================================
import os
output_dir = "SVE results"
os.makedirs(output_dir, exist_ok=True)
# Loss curve comparison
# Create comparison figure: ground truth and loss curves (top and bottom)
fig = plt.figure(figsize=(9, 9), dpi=150)    # Adjust to square layout
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

# Ground truth concentration field (top panel)
ax0 = fig.add_subplot(gs[0])

# cmap=colormap_parula,
h0 = ax0.imshow(Q, cmap=colormap_parula,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
ax0.set_title('(a) Ground Truth Concentration Field', fontsize=18)
ax0.set_ylabel('Temporal Domain ($t$)', fontsize=18)
ax0.tick_params(axis='both', which='major', labelsize=18)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(h0, cax=cax)
cbar.set_label('Concentration', fontsize=18, rotation=270)
cbar.ax.yaxis.set_label_coords(3.2, 0.5)
cbar.ax.tick_params(labelsize=18)
# fig.colorbar(h0, cax=cax, label='Concentration')

# Loss curve comparison (bottom panel)
ax1 = fig.add_subplot(gs[1])
ax1.semilogy(mlp_losses, label='MLP Total Loss', linewidth=1)
#ax1.semilogy(mlp_u_errors, '--', label='MLP U Loss', linewidth=1)
#ax1.semilogy(mlp_f_errors, ':', label='MLP F Loss', linewidth=1)
ax1.semilogy(kan_losses, label='KAN Total Loss', linewidth=1)
#ax1.semilogy(kan_u_errors, '--', label='KAN U Loss', linewidth=1)
#ax1.semilogy(kan_f_errors, ':', label='KAN F Loss', linewidth=1)
ax1.set_title('(b) Training Loss Curves', fontsize=18)
ax1.set_xlabel('Iterations', fontsize=18)
ax1.set_ylabel('Loss Value (log scale)', fontsize=18)
ax1.legend(fontsize=18, loc='upper right',frameon=False)
# ax1.grid(True, alpha=0.5)
ax1.tick_params(axis='both', which='major', labelsize=18)

#plt.tight_layout()
plt.savefig(f"{output_dir}/sve_groundtruth_losses.png", 
           bbox_inches='tight', 
           dpi=700,
           facecolor='white')
plt.show()

# ======================================================================
kan_log_error = np.log10(kan_error_Q + 1e-10)
mlp_log_error = np.log10(mlp_error_Q + 1e-10)

# Create professional layout canvas
fig = plt.figure(figsize=(9, 9), dpi=150)    # Optimized golden ratio dimensions
gs = gridspec.GridSpec(2, 2,    # 2-row 2-column structure
                       width_ratios=[1, 0.05],    # Main figure to colorbar width ratio
                       height_ratios=[1, 1],       # Top and bottom subplot height ratio
                       wspace=0.1, hspace=0.22)    # Fine-tune spacing

# =========================================================================
# KAN error heatmap
# =========================================================================
ax0 = fig.add_subplot(gs[0, 0])

# cmap=colormap_parula,
h0 = ax0.imshow(kan_log_error, cmap=colormap_parula,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto', 
               vmin=-3.4, vmax=-0.1)
ax0.set(xlabel='',    # Hide x-axis label to avoid duplication
       ylabel='Temporal Domain ($t$)',
       title='(e) KAN Prediction Error')
ax0.xaxis.label.set_size(18)
ax0.yaxis.label.set_size(18)
ax0.title.set_size(18)
ax0.tick_params(axis='both', which='major', labelsize=18)
# =========================================================================
# MLP error heatmap
# =========================================================================
ax1 = fig.add_subplot(gs[1, 0])

# cmap=colormap_parula,
h1 = ax1.imshow(mlp_log_error, cmap=colormap_parula,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto',
               vmin=-3.4, vmax=-0.1)
ax1.set(xlabel='Spatial Domain ($x$)',
       ylabel='Temporal Domain ($t$)',
       title='(f) MLP Prediction Error')
ax1.xaxis.label.set_size(18)
ax1.yaxis.label.set_size(18)
ax1.title.set_size(18)
ax1.tick_params(axis='both', which='major', labelsize=18)
# =========================================================================
# Shared colorbar system
# =========================================================================
cax = fig.add_subplot(gs[:, 1])    # Span rows on the right side
cbar = plt.colorbar(h1, cax=cax)
cbar.set_label('log10(|Prediction Error|)', 
              rotation=270, 
              labelpad=25,
              fontsize=18)
cbar.ax.tick_params(labelsize=18) 

# =========================================================================
# Professional save settings
# =========================================================================
plt.savefig(f"{output_dir}/SVE error_heatmaps_pro.png", 
           bbox_inches='tight', 
           dpi=600,
           facecolor='white',
           transparent=False,
           metadata={'Description': 'Prediction error comparison between KAN and MLP models',
                     'Author': 'Your Name'})    # Add metadata
plt.show()

# ======================================================================
# Prediction visualization (independent colorbars)
fig = plt.figure(figsize=(9, 9), dpi=150)
# Modify grid layout: 2 rows (KAN/MLP), 2 columns (main/colorbar)
gs = gridspec.GridSpec(2, 2, 
                       width_ratios=[1, 0.05],    # Main figure width : colorbar width = 9:1
                       height_ratios=[1, 1],
                       wspace=0.1, hspace=0.22)    # Increase column spacing

# KAN prediction
ax0 = fig.add_subplot(gs[0, 0])

# cmap=colormap_parula,
h0 = ax0.imshow(kan_pred, cmap=colormap_parula,
                extent=[x.min(), x.max(), t.min(), t.max()],
                origin='lower', aspect='auto',
                vmin=kan_pred.min(), vmax=kan_pred.max())
ax0.set(xlabel='', 
        ylabel='Temporal Domain ($t$)',
        title='(c) KAN Prediction')
ax0.xaxis.label.set_size(18)
ax0.yaxis.label.set_size(18)
ax0.title.set_size(18)
ax0.tick_params(axis='both', which='major', labelsize=18)
ax0.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=2, alpha=0.5)

# MLP prediction
ax1 = fig.add_subplot(gs[1, 0])

# cmap=colormap_parula,
h1 = ax1.imshow(mlp_pred, cmap=colormap_parula,
                extent=[x.min(), x.max(), t.min(), t.max()],
                origin='lower', aspect='auto',
                vmin=mlp_pred.min(), vmax=mlp_pred.max())
ax1.set(xlabel='Spatial Domain ($x$)',
        ylabel='Temporal Domain ($t$)',
        title='(d) MLP Prediction')
ax1.xaxis.label.set_size(18)        # Although xlabel is empty, can still set
ax1.yaxis.label.set_size(18)        # Y-axis label font size
ax1.title.set_size(18)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.plot(X_u_train[:, 0], X_u_train[:, 1], 'kx', markersize=2, alpha=0.5)

# KAN-specific colorbar
cax0 = fig.add_subplot(gs[0, 1])    # First row, second column
cbar0 = plt.colorbar(h0, cax=cax0)
cbar0.set_label('Concentration', rotation=270, labelpad=25, fontsize=18)    # Slightly reduce font size
cbar0.ax.tick_params(labelsize=18)

# MLP-specific colorbar
cax1 = fig.add_subplot(gs[1, 1])    # Second row, second column
cbar1 = plt.colorbar(h1, cax=cax1)
cbar1.set_label('Concentration', rotation=270, labelpad=25, fontsize=18)
cbar1.ax.tick_params(labelsize=18)

# Save figure
plt.savefig(f"{output_dir}/case1_prediction_comparison_separate_cb.png", 
            bbox_inches='tight', 
            dpi=600,
            facecolor='white')
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
  MLP: {np.max(mlp_error_Q):.4e}
  KAN: {np.max(kan_error_Q):.4e}

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
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], 
                       hspace=0.36, wspace=0.25)

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
ax1.set_title(r'(b) $\log_{10}|Q_{KAN} - Q_{num}|$', fontsize=title_font)
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
ax3.set_title(r'(d) $\log_{10}|Q_{MLP} - Q_{num}|$', fontsize=title_font)
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
h4 = ax4.imshow(Q, cmap=pred_cmap,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
ax4.set_title('(e) Numerical solution', fontsize=title_font)
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
plt.savefig(f"{output_dir}/SVE_3x2_visualization.png", 
           bbox_inches='tight', 
           dpi=600, 
           facecolor='white')
# Save high-quality figure
plt.savefig(f"{output_dir}/SVE_3x2_visualization.pdf", 
           bbox_inches='tight', 
           dpi=600, 
           facecolor='white')
plt.show()
