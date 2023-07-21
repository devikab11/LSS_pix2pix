import numpy as np
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import os
import time
# j_i=0
# j_f=9050 #train loader
# p_i=9051 #test loader 
# p_f=9056


def log_transform(arr):
    # Create a copy of the original array
    transformed_arr = np.copy(arr)

    # Add a constant offset to shift the values to a positive range
    transformed_arr += abs(np.min(transformed_arr)) + 1e-9

    # Apply the transformation equation to the array
    transformed_arr = (2 * transformed_arr / (transformed_arr + 4)) - 1
    #transformed_arr = np.log10(np.copy(arr))


    #filtered_array = gaussian_filter(transformed_arr, sigma=0.5)

    return transformed_arr


j_i=0
j_f=6000#train loader
p_i=6000 #val loader 
p_f=6500  
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loads_arrays(in_val,fin_val,in_=True):
    loaded_array=[]
    loaded_list=[]
    for i in range(in_val,fin_val):
        k=i+1
        if in_==True:
            loaded_array=np.load('/home/s6debhat/IF_128/IF_128/IN_File_128_{0}.npy'.format(k))
            loaded_list.append(loaded_array)

        else:
            loaded_array=np.load('/home/s6debhat/FF_128/FF_128/FI_File_128_{0}.npy'.format(k))
            loaded_list.append(loaded_array)

    return loaded_list

s=time.time()       
INITIAL=loads_arrays(j_i,j_f,in_=True)
#INITIAL=INITIAL.to(device)
e=time.time()
loading_time=e-s
print("time for initial loading",loading_time)
s1=time.time()
FINAL=loads_arrays(j_i,j_f,in_=False)
e1=time.time()
loading_time1=e1-s1
print("time for final loading",loading_time1)
INITIAL_VAL=loads_arrays(p_i,p_f,in_=True)
FINAL_VAL=loads_arrays(p_i,p_f,in_=False)
    
transformed_arrays_IF = [log_transform(arr) for arr in INITIAL]
transformed_arrays_FF = [log_transform(arr) for arr in FINAL]
transformed_arrays_IF_val = [log_transform(arr) for arr in INITIAL_VAL]
transformed_arrays_FF_val = [log_transform(arr) for arr in FINAL_VAL]

X_train=torch.Tensor(np.array(transformed_arrays_IF))
Y_train=torch.Tensor(np.array(transformed_arrays_FF))
X_val=torch.Tensor(np.array(transformed_arrays_IF_val))
Y_val=torch.Tensor(np.array(transformed_arrays_FF_val))
train_set = TensorDataset(X_train, Y_train)
val_set = TensorDataset(X_val, Y_val)
batch_size=1
train_loader= torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
val_loader= torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=True)





