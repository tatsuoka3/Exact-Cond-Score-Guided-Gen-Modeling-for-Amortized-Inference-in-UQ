from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from functools import partial
from utils_DM import reverse_SDE, cond_score_post, make_folder, cond_alpha, cond_beta2, b, sigma, s1, s2, s3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class FN_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size=100):
        super(FN_Net, self).__init__()
        self.input = nn.Linear(input_dim, hid_size)
        self.fc1 = nn.Linear(hid_size, hid_size)
        self.output = nn.Linear(hid_size, output_dim)
    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x

### Parameters
N_samples =  5000
N_location_samples = 10
N_coeffs = 4
VAR_Y = 1e-5
VAR_U = 0.1
VAR_V = VAR_U
dim_v = N_location_samples
dim_u = N_coeffs
savedir = f".../PDE_example/"
make_folder(savedir)


### load coefficients and corresponding solution data at 10 locs
x_sample = torch.tensor(np.load(savedir + "/data/b_mn_samples_5000.npy"), dtype=torch.float32, device=DEVICE).reshape(N_samples,-1)
y_sample = torch.tensor(np.load(savedir + "/data/sampled_solutions_5000_10_locs.npy"), dtype=torch.float32, device=DEVICE).reshape(N_samples,N_location_samples) 

y_std = 0.01
noise = torch.randn(N_samples, N_location_samples, device=DEVICE) * y_std 
y_sample = y_sample * (1 + noise)
perm = torch.randperm(y_sample.shape[0], device=DEVICE)
cond_Y = y_sample[perm]
np.save(savedir + "cond_Y.npy", cond_Y.cpu().numpy())

### set diffusion model parameters
sample_U = x_sample
np.save(savedir + "sample_U.npy", sample_U.cpu().numpy())
sample_V = y_sample
np.save(savedir + "sample_V.npy", sample_V.cpu().numpy())

mean_U = torch.mean(sample_U, dim=0)
std_U = torch.std(sample_U, dim=0)
mean_V = torch.mean(sample_V, dim=0)
std_V = torch.std(sample_V, dim=0)
    
sample_U_normalized = (sample_U - mean_U) / std_U
np.save(savedir + "sample_U_normalized.npy", sample_U_normalized.cpu().numpy())
sample_V_normalized = (sample_V - mean_V) / std_V
np.save(savedir + "sample_V_normalized.npy", sample_V_normalized.cpu().numpy())
cond_Y_normalized = (cond_Y - mean_V) / std_V
np.save(savedir + "cond_Y_normalized.npy", cond_Y_normalized.cpu().numpy())
sample_X_normalized = torch.cat([sample_U_normalized,sample_V_normalized], dim=1)
    
gen_sample_size = N_samples
TIME_STEPS = 1000
batch_size = 1000
num_batches = gen_sample_size // batch_size

## generate normal samples
xT = torch.randn(gen_sample_size, dim_u + dim_v, device=DEVICE)
torch.save(xT, savedir + "xT_amortized.pt")
samples_regen_list = []

VAR_U_gen = torch.ones(dim_u, device=DEVICE, dtype=torch.float32) * VAR_U
VAR_V_gen = torch.ones(dim_v, device=DEVICE, dtype=torch.float32) * VAR_V
VAR_Y_gen = torch.ones(dim_v, device=DEVICE, dtype=torch.float32) * VAR_Y

for batch_idx in range(num_batches):

    x_T_batch = xT[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    cond_Y_batch = cond_Y_normalized[batch_idx * batch_size : (batch_idx + 1) * batch_size]

    score_normal_cond_batch = partial(
        cond_score_post,
        sample_U=sample_U_normalized,
        sample_V=sample_V_normalized,
        cond_Y=cond_Y_batch,
        var_U=VAR_U_gen,
        var_V=VAR_V_gen,
        var_Y=VAR_Y_gen
    )

    samples_batch = reverse_SDE(
        x_T=x_T_batch,
        time_steps=TIME_STEPS,
        drift_fun=b,
        diffuse_fun=sigma,
        score=score_normal_cond_batch,
        save_path=False
    )

    samples_regen_list.append(samples_batch)
    print(f"Batch {batch_idx+1}/{num_batches}.")

samples_regen = torch.cat(samples_regen_list, dim=0)
samples_regen[:, 0:dim_u] = (samples_regen[:, 0:dim_u] * std_U) + mean_U
samples_regen[:, dim_u : dim_u + dim_v] = (samples_regen[:, dim_u : dim_u + dim_v] * std_V) + mean_V

np.save(
    os.path.join(savedir, f"samples_regen_{gen_sample_size}_generated_samples.npy"),
    samples_regen.cpu().numpy()
)

## network training
LEARNING_RATE = 1e-3
n_neurons = 100
n_layers = 1
total_epochs = 20000

### load data
# cond_Y = torch.tensor(np.load(savedir + "cond_Y.npy"), device = DEVICE, dtype=torch.float32)
# xT = torch.load(savedir + "xT_amortized.pt")
# samples_regen = torch.tensor(np.load(savedir + "samples_regen_{gen_sample_size}_generated_samples.npy"), device = DEVICE, dtype=torch.float32)

FN = FN_Net(dim_u + dim_v + dim_v, dim_u, hid_size=n_neurons).to(DEVICE)
optimizer = optim.Adam(FN.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

yTrain = torch.hstack((cond_Y.reshape(-1, dim_v), xT[:, 0 : dim_u + dim_v].reshape(-1, dim_u + dim_v)))
xTrain = samples_regen[:, 0:dim_u].reshape(-1, dim_u)

xTrain_mean = xTrain.mean(dim=0, keepdim=True)
xTrain_std = xTrain.std(dim=0, keepdim=True)
yTrain_mean = yTrain.mean(dim=0, keepdim=True)
yTrain_std = yTrain.std(dim=0, keepdim=True)

xTrain_normalized = (xTrain - xTrain_mean) / xTrain_std
yTrain_normalized = (yTrain - yTrain_mean) / yTrain_std

training_loss = []
best_loss = float('inf')
best_state_dict = None
best_epoch = -1

model_save_path = os.path.join(savedir, f"FN_trained_model_seed_{SEED}.pth")
stats_save_path = os.path.join(savedir, f"FN_trained_model_seed_{SEED}_stats.pth")

for j in range(total_epochs):
    optimizer.zero_grad()
    pred = FN(yTrain_normalized)
    loss = criterion(pred, xTrain_normalized)
    training_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    # Track best model weights
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = j
        best_state_dict = {
            'model_state_dict': FN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': j,
            'best_loss': best_loss
        }

    if j % 100 == 0:
        print(f"Epoch {j}: Loss {loss.item()}")

torch.save(best_state_dict, model_save_path)
torch.save({
    'xTrain_mean': xTrain_mean,
    'xTrain_std': xTrain_std,
    'yTrain_mean': yTrain_mean,
    'yTrain_std': yTrain_std
}, stats_save_path)
training_loss = np.array(training_loss)
epochs = np.arange(0, total_epochs, 1)
plt.plot(epochs, training_loss, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig(savedir + 'training_loss.png')
plt.show()

### reload network
FN = FN_Net(dim_u + dim_v + dim_v, dim_u).to(DEVICE)
optimizer_loaded = optim.Adam(FN.parameters(), lr=LEARNING_RATE)

# model_save_path = os.path.join(savedir, f"FN_trained_model_seed_{SEED}.pth")
# stats_save_path = os.path.join(savedir, f"FN_trained_model_seed_{SEED}_stats.pth")

checkpoint = torch.load(model_save_path, map_location=DEVICE)
FN.load_state_dict(checkpoint['model_state_dict'])
optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])
    
stats = torch.load(stats_save_path, map_location=DEVICE)
xTrain_mean = stats['xTrain_mean']
xTrain_std  = stats['xTrain_std']
yTrain_mean = stats['yTrain_mean']
yTrain_std  = stats['yTrain_std']

    
### load test case 1 or 2  
# y_obs = np.load(savedir + "/data/testIC_1.npy").reshape(1,dim_v) 
y_obs = np.load(savedir + "/data/testIC_2.npy").reshape(1,dim_v) 

n_gaussian_samples = 5000
y_sample = y_obs * np.ones((n_gaussian_samples,dim_v))
zT = np.random.randn(n_gaussian_samples, dim_u + dim_v)
yTrain = torch.tensor(np.hstack((y_sample, zT)), dtype=torch.float32).to(DEVICE)
output = (FN((yTrain-yTrain_mean)/yTrain_std)*xTrain_std + xTrain_mean).detach().cpu().numpy()
output = output.reshape(n_gaussian_samples, dim_u) 
np.save(savedir +"NN_output_testIC_2.npy", output)

### generate corresponding solution data
VAR_U = 0.1 
mesh = UnitSquareMesh(31, 31)
V = FunctionSpace(mesh, "P", 2)
grid_N = 32
x_vals = np.linspace(0, 1, grid_N)
y_vals = np.linspace(0, 1, grid_N)
X, Y = np.meshgrid(x_vals, y_vals)
sampling_points = np.stack([X, Y], axis=-1)

N_random_samples = 10
M_modes = 2
N_coeffs = 4
n_samples = 5000
    
sampled_solutions = np.zeros((n_samples, 32, 32))
b_mn_samples = np.load(savedir + "NN_output_testIC_2.npy")[:,0:N_coeffs]
b_mn_samples = b_mn_samples.reshape(n_samples,M_modes,M_modes)
    
class KLEExpression(UserExpression):
    def __init__(self, coeffs, M, **kwargs):
        self.coeffs = coeffs
        self.M = M
        super().__init__(**kwargs)
    
    def eval(self, values, x):
        result = 0.0
        for m in range(1, self.M + 1):
            for n in range(1, self.M + 1):
                b = self.coeffs[m - 1, n - 1]
                result += b * np.sin(m * 2 * np.pi * x[0]) * np.sin(n * 2 * np.pi * x[1])
        values[0] = np.exp(result)
    
    def value_shape(self):
        return ()
        
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)
L = f * v * dx
bc = DirichletBC(V, Constant(0.0), 'on_boundary')
    
for k in range(n_samples):
    kle_expr = interpolate(KLEExpression(b_mn_samples[k], M_modes, degree=2), V)
    A = assemble(kle_expr * dot(grad(u), grad(v)) * dx)
    b = assemble(L)
    bc.apply(A, b)
    u_sol = Function(V)
    solve(A, u_sol.vector(), b)
    flat_sampling_points = sampling_points.reshape(-1, 2)
    u_sampled = [u_sol(pt) for pt in flat_sampling_points]
    sampled_solutions[k, :, :] = np.array(u_sampled).reshape(32, 32)

np.save(savedir + "generated_solutions_test_IC_2.npy", sampled_solutions)










