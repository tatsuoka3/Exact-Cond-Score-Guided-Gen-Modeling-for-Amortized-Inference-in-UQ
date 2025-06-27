import numpy as np
import torch
import os

EPS_ALPHA = 0.00001
EPS_BETA = 0.00001

def cond_alpha(t):
    return 1 - (1 - EPS_ALPHA) * t

def cond_beta2(t):
    return EPS_BETA + (1 - EPS_BETA) * t

def b(t):
    return -(1 - EPS_ALPHA) / cond_alpha(t)

def sigma_sq(t):
    return (1 - EPS_BETA) - 2 * b(t) * cond_beta2(t)

def sigma(t):
    return np.sqrt(sigma_sq(t))
    
def s1(t, VAR_0):
    return cond_beta2(t) / (cond_alpha(t)**2 * VAR_0 + cond_beta2(t))

def s2(t, VAR_0):
    return (cond_alpha(t)*VAR_0) / (cond_alpha(t)**2 * VAR_0 + cond_beta2(t))

def s3(t, VAR_0):
    return (cond_beta2(t)*VAR_0) / (cond_alpha(t)**2 * VAR_0 + cond_beta2(t))

def reverse_SDE(x_T, time_steps, drift_fun, diffuse_fun, score, save_path=True):
    # x0: (N, d)
    N = x_T.shape[0]
    d = x_T.shape[1]
    # Generate the time mesh
    dt = 1.0/time_steps
    # Initialization
    xt = x_T.clone()
    t = 1.0
    # define storage
    t_vec = [t]
    path_all = [xt]

    # forward Euler
    for i in range(time_steps):
        # Evaluate the diffusion term
        diffuse = diffuse_fun(t)
        # Evaluate the drift term
        drift = drift_fun(t)*xt - diffuse**2 * score(xt, t)/2
        # Update
        xt = xt - dt*drift
        # Store the state in the path
        if save_path:
            path_all.append(xt)
        t_vec.append(t)
        # update time
        t = t - dt
    if save_path:
        return path_all, t_vec
    else:
        return xt
        
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
    m = zt[:, None, dim_u:] * s2(t, var_V) + sample_V[None, : , :  ] * s1(t, var_V)
    y_diff = zt[:, None, dim_u:] * s2(t, var_V) + sample_V[None, : , :  ] * s1(t, var_V)  - cond_Y[:, None, :] # (N_eval, N_sample,  dim_v)
    # y_diff = zt[:, None, dim_u:] * s2(t, var_V) + sample_V[None, : , :  ] * s1(t, var_V)  - cond_Y[:, None, :] # (N_eval, N_sample,  dim_v)
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

def make_folder(folder):
    """Create the folder if it doesn't exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        print(f"Folder '{folder}' already exists.")
    