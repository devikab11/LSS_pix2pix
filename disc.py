#Discriminator
import numpy as np
#import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Conv3DTranspose
#from keras.optimizers import Adam
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#import cv2
#import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import os

def cnn_block(in_channels,out_channels,kernel_size,stride,padding, first_layer = False):

   if first_layer:
       return nn.Conv3d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
   else:
       return nn.Sequential(
           nn.Conv3d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
           nn.BatchNorm3d(out_channels,momentum=0.1,eps=1e-5),
           )

def tcnn_block(in_channels,out_channels,kernel_size,stride,padding,output_padding=0, first_layer = False):
   if first_layer:
       return nn.ConvTranspose3d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding)

   else:
       return nn.Sequential(
           nn.ConvTranspose3d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
           nn.BatchNorm3d(out_channels,momentum=0.1,eps=1e-5),
           )
   
#parameters

gf_dim = 64
df_dim = 64
in_w = in_h = in_d = 128 #dimensions of a single data sample
c_dim = 1 #color channels, 1 for grayscale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
 def __init__(self,instance_norm=False):#input : 1 x 128^3
   super(Discriminator,self).__init__()
   self.conv1 = cnn_block(c_dim*2,df_dim,4,2,1, first_layer=True) # 64 x 64^3  #c_dim x 2 because you are giving it x and y both!! concatenated along the channel
   self.conv2 = cnn_block(df_dim,df_dim*2,4,2,1)# 128 x 32^3
   self.conv3 = cnn_block(df_dim*2,df_dim*4,4,2,1)# 256 x 16^3
   self.conv4 = cnn_block(df_dim*4,df_dim*8,4,1,1)# 512 x 15^3
   self.conv5 = cnn_block(df_dim*8,1,4,1,1, first_layer=True)# 512 x 14^3

   self.sigmoid = nn.Sigmoid()
 def forward(self, x, y):
   O = torch.cat([x,y],dim=1)
   O = F.leaky_relu(self.conv1(O),0.2)
   O = F.leaky_relu(self.conv2(O),0.2)
   O = F.leaky_relu(self.conv3(O),0.2)
   O = F.leaky_relu(self.conv4(O),0.2)
   O = self.conv5(O)

  # return self.sigmoid(O)
   return O


def test_d():
  x= torch.randn(1,1,128,128,128)
  y= torch.randn(1,1,128,128,128)
  model=Discriminator()
  preds=model(x,y)
  print(preds.shape)

#test_d()
