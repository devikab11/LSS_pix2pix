#save some examples

import numpy as np
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import os
torch.manual_seed(545667)
workers = 2

epochs = 51

gf_dim = 64
df_dim = 64

L1_lambda = 100.0

in_w = in_h = in_d = 64 #dimensions of a single data sample
c_dim = 1 #color channels, 1 for grayscale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("dataset_loader.py") as f:
    exec(f.read())

with open("gen.py") as f:
    exec(f.read())

with open("disc.py") as f:
    exec(f.read())

with open("train_July4.py") as f:
    exec(f.read())


print("mission successful")

# os.makedirs('savedsamples_June27', exist_ok=True)
# def save_samples(generator_state_dict, test_loader):
#     G = Generator()
#     G.load_state_dict(torch.load(generator_state_dict))

#     for i, data in enumerate(test_loader):
#         x, y = data
#         x = x.unsqueeze(1)
#         y = y.unsqueeze(1)

        
#         generated_samples = G(x)
#         print(generated_samples.size())

#         x = generated_samples[0]
#         #x_size = x.size()
#         k = x.squeeze(0)
#         print(k.size())
#         #k_size = k.size()
#         k1 = k.detach().numpy()

#         np.save(f'savedsamples_June27/generated_sample{i}.npy', k1)
#         np.save(f'savedsamples_June27/ground_truth{i}.npy', y.numpy())
#         np.save(f'savedsamples_June27/initial_sample{i}.npy', x.numpy())

#     print("Done!")


