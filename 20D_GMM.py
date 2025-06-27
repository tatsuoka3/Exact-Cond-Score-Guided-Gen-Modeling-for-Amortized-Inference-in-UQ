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


N_sample = 150000
N_gen = 30000

dim_u = 15
dim_v = 5

VAR_U = 0.1
VAR_V = VAR_U
VAR_Y = 1e-5 

savedir = ".../20D_GMM_example/"
make_folder(savedir)

### generate data
D = 20
sigma_value = 1.0
d_target = 4.0
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

sample_U = samples[:,0:dim_u]
sample_V = samples[:,dim_u:D]
np.save(savedir + "sample_U.npy", sample_U.cpu().numpy())
np.save(savedir + "sample_V.npy", sample_V.cpu().numpy())

mean_U = torch.mean(sample_U, dim=0)
std_U = torch.std(sample_U, dim=0)
mean_V = torch.mean(sample_V, dim=0)
std_V = torch.std(sample_V, dim=0)
    
sample_U_normalized = (sample_U - mean_U) / std_U
np.save(savedir + "sample_U_normalized.npy", sample_U_normalized.cpu().numpy())
sample_V_normalized = (sample_V - mean_V) / std_V
np.save(savedir + "sample_V_normalized.npy", sample_V_normalized.cpu().numpy())

filtered_conds = torch.zeros(dim_v, device=DEVICE)
cond_Y = get_samples(sample_V, filtered_conds, n_closest=30000)
np.save(savedir + "cond_Y.npy", cond_Y.cpu().numpy())

cond_Y_normalized = (cond_Y - mean_V) / std_V
np.save(savedir + "cond_Y_normalized.npy", cond_Y_normalized.cpu().numpy())

xT = torch.randn(N_gen, dim_u + dim_v, device=DEVICE, dtype=torch.float32)
np.save(savedir + "xT_amortized.npy", xT.cpu().numpy())

TIME_STEPS = 1000
score_batch_size = 150
num_batches = N_gen // score_batch_size

VAR_V = VAR_U
VAR_Y = 1e-5

VAR_U_gen = torch.ones(dim_u, device=DEVICE, dtype=torch.float32) * VAR_U
VAR_V_gen = torch.ones(dim_v, device=DEVICE, dtype=torch.float32) * VAR_V
VAR_Y_gen = torch.ones(dim_v, device=DEVICE, dtype=torch.float32) * VAR_Y
samples_regen_list = []
    
for batch_idx in range(num_batches):
    print(batch_idx)
    x_T_batch = xT[batch_idx * score_batch_size : (batch_idx + 1) * score_batch_size]
    cond_Y_batch = cond_Y[batch_idx * score_batch_size : (batch_idx + 1) * score_batch_size]
    score_normal_cond_batch = partial(
        cond_score_post,
        sample_U=sample_U,
        sample_V=sample_V,
        cond_Y=cond_Y_batch,
        var_U=VAR_U_gen,
        var_V=VAR_V_gen,
        var_Y=VAR_Y_gen)
    
    samples_batch = reverse_SDE(
        x_T=x_T_batch,
        time_steps=TIME_STEPS,
        drift_fun=b,
        diffuse_fun=sigma,
        score=score_normal_cond_batch,
        save_path=False)
    samples_regen_list.append(samples_batch)
    print(f"Batch {batch_idx + 1}/{num_batches} completed.")
        
samples_regen = torch.cat(samples_regen_list, dim=0)
np.save(savedir + f"samples_regen.npy", samples_regen.cpu().numpy())

### train neural network
LEARNING_RATE = 1e-3
n_neurons = 50
n_layers = 2
total_epochs = 50000

### load data
cond_Y = torch.tensor(np.load(savedir + "cond_Y.npy"), device = DEVICE, dtype=torch.float32)
xT = torch.tensor(np.load(savedir + "xT_amortized.npy"), device = DEVICE, dtype=torch.float32)
samples_regen = torch.tensor(np.load(savedir + "samples_regen.npy"), device = DEVICE, dtype=torch.float32)
    
FN = FN_Net(dim_u + dim_v + dim_v, dim_u, hid_size = n_neurons).to(DEVICE)
optimizer = optim.Adam(FN.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
yTrain = torch.hstack((cond_Y.reshape(-1, dim_v), xT.reshape(-1, dim_u + dim_v))) 
xTrain = samples_regen[:, 0:dim_u].reshape(-1, dim_u) 
training_loss = []

### save best model
best_loss = float('inf')
best_epoch = -1
best_state_dict = None

### training loop
for j in range(total_epochs):
    optimizer.zero_grad()
    pred = FN(yTrain)
    loss = criterion(pred, xTrain)
    training_loss.append(loss.item())  
    loss.backward()
    optimizer.step()

    # Save best model
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

model_save_path = os.path.join(savedir, f"FN_trained_model_seed_{SEED}.pth")
torch.save(best_state_dict, model_save_path)
training_loss = np.array(training_loss)
epochs = np.arange(0, total_epochs, 1)
plt.plot(epochs, training_loss, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.savefig(savedir + 'training_loss.png')
plt.show()
print(f"Best model saved to {model_save_path}, epoch {best_epoch}, loss {best_loss:.6f}")

### reload network
FN = FN_Net(dim_u + dim_v + dim_v, dim_u, hid_size = n_neurons).to(DEVICE)
optimizer_loaded = optim.Adam(FN.parameters(), lr=LEARNING_RATE)
# model_save_path = os.path.join(savedir, f"FN_trained_model_seed_{SEED}.pth")
checkpoint = torch.load(model_save_path, map_location=DEVICE)
FN.load_state_dict(checkpoint['model_state_dict'])
optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])

### run test case
condition = -0.5
n_gaussian_samples = 5000
y_sample = condition * np.ones((n_gaussian_samples,dim_v))
zT = np.random.randn(n_gaussian_samples, dim_u+dim_v)
yTrain = torch.tensor(np.hstack((y_sample, zT)), dtype=torch.float32).to(DEVICE)
output = FN(yTrain).detach().cpu().numpy()
output = output.reshape(n_gaussian_samples, dim_u) 
np.save(savedir + f"NN_output_testIC_{condition}.npy", output)
    









