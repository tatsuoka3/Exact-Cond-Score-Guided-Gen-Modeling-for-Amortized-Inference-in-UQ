import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
from utils_DM import reverse_SDE, make_folder, cond_alpha, cond_beta2, b, sigma, s1, s2, s3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 1234
torch.manual_seed(SEED) 
np.random.seed(SEED)

def cond_score_post(zt, t, sample_U, sample_V, cond_Y, var_U, var_V, var_Y):
    """
    Compute the score of the conditional distribution of Z0 given Y
    zt: (N_eval, dim_u + dim_v)
    t: time (scalar)
    sample_U: (N_sample, dim_u)
    sample_V: (N_sample, dim_v)
    cond_Y: (dim_v)
    var_U: (dim_u)
    var_V: (dim_v)
    var_Y: (dim_v)
    """
    dim_u = sample_U.shape[1]
    dim_v = sample_V.shape[1]
    
    joint_sample = torch.cat([sample_U, sample_V], dim=1) # (N_sample, dim_u + dim_v)
    var_joint = torch.cat([var_U, var_V], dim=0) # (dim_u + dim_v)
    # forward kernel
    mu_t = joint_sample * cond_alpha(t) # (N_sample, dim_u + dim_v)
    var_t = cond_beta2(t) + cond_alpha(t) ** 2 * var_joint # (dim_u + dim_v)
    # score calculation
    x_diff = zt[:, None, :] - mu_t[None, :, :] # (N_eval, N_sample, dim_u + dim_v)
    # prior Gaussian score
    score =  - x_diff / var_t # (N_eval, N_sample, dim_u + dim_v)
    # get score weight: q(xi| zt, y) = q(xi|zt) q(y|zt,xi)
    # prior weight: q(xi|zt)
    log_pt =  0.5 * torch.sum(score * x_diff, dim=2) # (N_eval, N_sample)
    # add q(y|zt,xi)
    # cond weighting function
    # mu_v - y
    y_diff = zt[:, None, dim_u:] * s2(t, var_V) + sample_V[None, : , :  ] * s1(t, var_V)  - cond_Y[None, None, :] # (N_eval, N_sample,  dim_v)
    # likelihood score: J(t) * E[S(Z0) | Y,Zt]
    score_likelihood = - y_diff / (s3(t, var_V) + var_Y) # (N_eval, N_sample, dim_v)
    # final score unweighted (N_eval, N_sample, dim_u + dim_v)
    score[:, :, dim_u:] = score[:, :, dim_u:] + score_likelihood * s2(t, var_V) 
    #  cond weighting function
    cond_weight_log = 0.5 * torch.sum(score_likelihood * y_diff, dim=2) # (N_eval, N_sample)
    log_pt += cond_weight_log
    # score weight
    # numerical normalization
    log_pt = log_pt - torch.max(log_pt, dim=1, keepdim=True)[0] # (N_eval, N_sample)
    pt = torch.exp(log_pt) # (N_eval, N_sample)
    wt = pt / torch.sum(pt, dim=1, keepdim=True) # (N_eval, N_sample)
    # weighted final score
    score_final = torch.sum(score * wt[:, :, None], dim=1) # (N_eval, dim_x)
    return score_final
    

savedir = '.../bimodal_example/'
make_folder(savedir)

### generate data
N_sample = 5000
dim_u = 1
dim_v = 1 
u_max = 4
sample_U = -2 + torch.rand(N_sample, dim_u, device=DEVICE, dtype=torch.float32)*u_max
y_std = np.sqrt(0.1)
sample_V = sample_U**2 + torch.randn(N_sample, dim_v, device=DEVICE, dtype=torch.float32) * y_std
cond_Y = 1

###normalize data
mean_U = torch.mean(sample_U, dim=0)
std_U = torch.std(sample_U, dim=0)
mean_V = torch.mean(sample_V, dim=0)
std_V = torch.std(sample_V, dim=0)
sample_U_normalized = (sample_U - mean_U) / std_U
sample_V_normalized = (sample_V - mean_V) / std_V
cond_Y_normalized = (cond_Y - mean_V) / std_V

### diffusion model parameters
TIME_STEPS = 1000
VAR_Y = 1e-4
VAR_U = 0.005
VAR_V = VAR_U

N_gen = 10000
x_T = torch.randn(N_gen, dim_u + dim_v, device=DEVICE, dtype=torch.float32)

VAR_U_gen = torch.ones(dim_u, device=DEVICE, dtype=torch.float32) * VAR_U
VAR_V_gen = torch.ones(dim_u, device=DEVICE, dtype=torch.float32) * VAR_V
VAR_Y_gen = torch.ones(dim_v, device=DEVICE, dtype=torch.float32) *  VAR_Y
         
score_normal_cond = partial(cond_score_post, sample_U=sample_U_normalized, sample_V=sample_V_normalized, cond_Y=cond_Y_normalized,var_U=VAR_U_gen, var_V=VAR_V_gen, var_Y=VAR_Y_gen)
samples_regen = reverse_SDE(x_T=x_T, time_steps=TIME_STEPS, drift_fun=b, diffuse_fun=sigma, score=score_normal_cond, save_path=False)
samples_regen[:, 0] = (samples_regen[:, 0] * std_U) + mean_U
samples_regen[:, 1] = (samples_regen[:, 1] * std_V) + mean_V
np.save(os.path.join(savedir, "samples_regen.npy"), samples_regen.cpu().numpy())

plt.hist(samples_regen[:, 0].cpu().numpy().flatten(), bins=50, density=True, alpha=0.4, color='blue', label='Diffusion Model',histtype='stepfilled')
plt.xlabel("$U$", fontsize= 15)
plt.ylabel("Density", fontsize= 15)
plt.ylim([0, 1.4])
plt.xlim([-3, 3])
plt.savefig(savedir + f'varU_{VAR_U}_varV_{VAR_V}_varY_{VAR_Y}.png')
plt.show()


