import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from functools import partial
from utils_DM import reverse_SDE, cond_score_post, make_folder, cond_alpha, cond_beta2, b, sigma, s1, s2, s3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

def sample_from_gmm(means, cov, weights, n_samples):
    n_components = len(means)
    labels = np.random.choice(n_components, size=n_samples, p=weights)
    samples = np.array([
        np.random.multivariate_normal(mean=means[k], cov=cov)
        for k in labels
    ])
    return samples, labels

### function for retrieving subset of v samples from the N_sample set for defining cond_Y in diffusion model (subselect N_gen samples closest to the 20D zero vector)
def get_samples(sample_V, target_vec, n_closest=10000):
    diffs = sample_V - target_vec[None, :]
    dists = torch.norm(diffs, dim=1)
    closest_indices = torch.topk(dists, k=n_closest, largest=False).indices
    return sample_V[closest_indices]

class FN_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hid_size=50):
        super(FN_Net, self).__init__()
        self.input = nn.Linear(input_dim, hid_size)
        self.fc1 = nn.Linear(hid_size, hid_size)
        self.fc2 = nn.Linear(hid_size, hid_size)
        self.output = nn.Linear(hid_size, output_dim)
    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.output(x)
        return x
        
        
LEARNING_RATE = 1e-3
n_neurons = 50
n_layers = 2
total_epochs = 50000

N_sample = 150000
N_gen = 30000

dim_u = 10
dim_v = 10


VAR_U = 0.2
VAR_V = VAR_U
VAR_Y = 1e-5

###test cases
conditions = [-0.5, 0.0, 0.5]
n_gaussian_samples = 5000

savedir = f".../"
make_folder(savedir)

###generate data
D = 20
sigma_value = 1.0
weights_full = [0.5, 0.5]

D_split1 = [0, 1, 2, 3, 4]
D_split2 = [5, 6, 7, 8, 9]
D_split3 = [10, 11, 12, 13, 14]
D_split4 = [15, 16, 17, 18, 19]

delta1 = 1.35
delta2 = 0.5
delta3 = 0.2
delta4 = 0.1

mu1 = np.zeros(D)
mu2 = np.zeros(D)
mu1[D_split1] = -delta1
mu2[D_split1] = +delta1
mu1[D_split2] = -delta2
mu2[D_split2] = +delta2
mu1[D_split3] = -delta3
mu2[D_split3] = +delta3
mu1[D_split4] = -delta4
mu2[D_split4] = +delta4

cov = sigma_value**2 * np.eye(D)
samples, _ = sample_from_gmm([mu1, mu2], cov, weights_full, n_samples=N_sample)
perm = np.random.permutation(samples.shape[0])
samples = torch.tensor(samples[perm], device=DEVICE, dtype=torch.float32)

sample_U = samples[:, 0:dim_u]
sample_V = samples[:, dim_u:D]

np.save(savedir + "sample_U.npy", sample_U.detach().cpu().numpy())
np.save(savedir + "sample_V.npy", sample_V.detach().cpu().numpy())

mean_U = torch.mean(sample_U, dim=0)
std_U  = torch.std(sample_U, dim=0)
mean_V = torch.mean(sample_V, dim=0)
std_V  = torch.std(sample_V, dim=0)

np.save(savedir + "mean_U.npy", mean_U.detach().cpu().numpy())
np.save(savedir + "std_U.npy",  std_U.detach().cpu().numpy())
np.save(savedir + "mean_V.npy", mean_V.detach().cpu().numpy())
np.save(savedir + "std_V.npy",  std_V.detach().cpu().numpy())

### standardized data
sample_Uz = ((sample_U - mean_U) / std_U)
sample_Vz = ((sample_V - mean_V) / std_V)

np.save(savedir + "sample_U_normalized.npy", sample_Uz.detach().cpu().numpy())  # now standardized
np.save(savedir + "sample_V_normalized.npy", sample_Vz.detach().cpu().numpy())  # now standardized


###select N_gen samples from sample_Vz
filtered_conds = torch.zeros(dim_v, device=DEVICE)
cond_Y = get_samples(sample_Vz, filtered_conds, n_closest=N_gen)
np.save(savedir + "cond_Y_normalized.npy", cond_Y.detach().cpu().numpy())  # standardized cond_Y

########## diffusion sampling in standardized space (commenr out if have samples) ###########

xT = torch.randn(N_gen, dim_u + dim_v, device=DEVICE, dtype=torch.float32)
np.save(savedir + "xT_amortized.npy", xT.detach().cpu().numpy())

TIME_STEPS = 1000
score_batch_size = 150
num_batches = (N_gen + score_batch_size - 1) // score_batch_size

VAR_Uz = (VAR_U / (std_U ** 2)).to(DEVICE).float()
VAR_Vz = (VAR_V / (std_V ** 2)).to(DEVICE).float()
VAR_Yz = (VAR_Y / (std_V ** 2)).to(DEVICE).float()

samples_regen_list = []
with torch.no_grad():
    for batch_idx in range(num_batches):
        i0 = batch_idx * score_batch_size
        i1 = min((batch_idx + 1) * score_batch_size, N_gen)

        x_T_batch = xT[i0:i1]
        cond_Y_batch = cond_Y[i0:i1]

        score_normal_cond_batch = partial(
            cond_score_post,
            sample_U=sample_Uz,
            sample_V=sample_Vz,
            cond_Y=cond_Y_batch,
            var_U=VAR_Uz,
            var_V=VAR_Vz,
            var_Y=VAR_Yz
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
        print(f"Batch {batch_idx + 1}/{num_batches} completed.")

samples_regen_normalized = torch.cat(samples_regen_list, dim=0)  # standardized samples (Uz,Vz)
np.save(savedir + "samples_regen_normalized.npy", samples_regen_normalized.detach().cpu().numpy())

Uz = samples_regen_normalized[:, :dim_u]
Vz = samples_regen_normalized[:, dim_u:]
U_phys = Uz * std_U + mean_U
V_phys = Vz * std_V + mean_V
samples_regen = torch.cat([U_phys, V_phys], dim=1)
np.save(savedir + "samples_regen.npy", samples_regen.detach().cpu().numpy())

########## train NN in standardized space (comment out if have trained model) ###########

# load standardized data
cond_Y = torch.tensor(np.load(savedir + "cond_Y_normalized.npy"), device=DEVICE, dtype=torch.float32)
xT = torch.tensor(np.load(savedir + "xT_amortized.npy"), device=DEVICE, dtype=torch.float32)
samples_regen_normalized = torch.tensor(np.load(savedir + "samples_regen_normalized.npy"), device=DEVICE, dtype=torch.float32)

FN = FN_Net(dim_u + dim_v + dim_v, dim_u, hid_size=n_neurons).to(DEVICE)
optimizer = optim.Adam(FN.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

yTrain = torch.hstack((cond_Y.reshape(-1, dim_v), xT.reshape(-1, dim_u + dim_v)))
xTrain = samples_regen_normalized[:, 0:dim_u].reshape(-1, dim_u)  # predicts Uz

EPS = 1e-8
y_mean = yTrain.mean(dim=0, keepdim=True)
y_std  = yTrain.std(dim=0, keepdim=True).clamp_min(EPS)

x_mean = xTrain.mean(dim=0, keepdim=True)
x_std  = xTrain.std(dim=0, keepdim=True).clamp_min(EPS)

yTrain_n = (yTrain - y_mean) / y_std
xTrain_n = (xTrain - x_mean) / x_std

training_loss = []
best_loss = float('inf')
best_epoch = -1
best_state_dict = None

for j in range(total_epochs):
    optimizer.zero_grad()

    pred_n = FN(yTrain_n)
    loss = criterion(pred_n, xTrain_n)

    training_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = j
        best_state_dict = {
            'model_state_dict': FN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': j,
            'best_loss': best_loss,
            'y_mean': y_mean.detach().cpu(),
            'y_std':  y_std.detach().cpu(),
            'x_mean': x_mean.detach().cpu(),
            'x_std':  x_std.detach().cpu(),
        }

    if j % 100 == 0:
        print(f"Epoch {j}: Loss {loss.item()}")

model_save_path = os.path.join(savedir, f"FN_trained_model_seed_{SEED}.pth")
torch.save(best_state_dict, model_save_path)

training_loss = np.array(training_loss)
epochs = np.arange(0, total_epochs, 1)
plt.plot(epochs, training_loss, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (normalized MSE)")
plt.title("Training Loss")
plt.legend()
plt.savefig(savedir + 'training_loss_normalized.png')
plt.show()
print(f"Best model saved to {model_save_path}, epoch {best_epoch}, loss {best_loss:.6f}")

########### reload network for test conditions ###########
FN = FN_Net(dim_u + dim_v + dim_v, dim_u, hid_size=n_neurons).to(DEVICE)
optimizer_loaded = optim.Adam(FN.parameters(), lr=LEARNING_RATE)
model_save_path = os.path.join(savedir, f"FN_trained_model_seed_{SEED}.pth")
checkpoint = torch.load(model_save_path, map_location=DEVICE)
FN.load_state_dict(checkpoint['model_state_dict'])
optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])

y_mean = checkpoint['y_mean'].to(DEVICE)
y_std  = checkpoint['y_std'].to(DEVICE)
x_mean = checkpoint['x_mean'].to(DEVICE)
x_std  = checkpoint['x_std'].to(DEVICE)
 
mean_V = np.load(savedir + "mean_V.npy")
std_V  = np.load(savedir + "std_V.npy")

mean_U = np.load(savedir + "mean_U.npy")
std_U = np.load(savedir + "std_U.npy")

for condition in conditions:

    y = condition * np.ones((n_gaussian_samples, dim_v))
    y_normalized = (y - mean_V[None, :]) / std_V[None, :]
    zT = np.random.randn(n_gaussian_samples, dim_u + dim_v)
    yTest = torch.tensor(np.hstack((y_normalized, zT)), dtype=torch.float32).to(DEVICE)

    yTest_n = (yTest - y_mean) / y_std
    out_n = FN(yTest_n)
    # NN output
    Uz_pred = ((out_n * x_std + x_mean)
               .detach().cpu().numpy().reshape(n_gaussian_samples, dim_u))
    # map to data space
    U_pred = Uz_pred * std_U[None, :] + mean_U[None, :]
    np.save(savedir + f"NN_output_testIC_{condition}_{VAR_U}_VAR_U_{dim_u}_dim_u.npy", U_pred)


