#generator
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

#generator

class Generator(nn.Module):

 def __init__(self,instance_norm=False):#input : 1 x 128^3
   super(Generator,self).__init__()
   self.e1 = cnn_block(c_dim,gf_dim,4,2,1, first_layer = True) #64 x 64^3
   self.e2 = cnn_block(gf_dim,gf_dim*2,4,2,1) #128 x 32^3
   self.e3 = cnn_block(gf_dim*2,gf_dim*4,4,2,1) #256 x 16^3
   self.e4 = cnn_block(gf_dim*4,gf_dim*8,4,2,1) #512 x 8^3
   self.e5 = cnn_block(gf_dim*8,gf_dim*8,4,2,1) #512 x 4^3
   self.e6 = cnn_block(gf_dim*8,gf_dim*8,4,2,1) #512 x 2^3
   self.e7 = cnn_block(gf_dim*8,gf_dim*8,4,2,1,first_layer=True) #512 x 1^3 #bottleneck

   self.d1 = tcnn_block(gf_dim*8,gf_dim*8,4,2,1) # 512 x 2^3
   self.d2 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1) #512 x 4^3
   self.d3 = tcnn_block(gf_dim*8*2,gf_dim*8,4,2,1) #512 x 8^3
   self.d4 = tcnn_block(gf_dim*8*2,gf_dim*4,4,2,1) #256 x 16^3
   self.d5 = tcnn_block(gf_dim*4*2,gf_dim*2,4,2,1) #128 x 32^3
   self.d6 = tcnn_block(gf_dim*2*2,gf_dim*1,4,2,1) #64 x 64^3
   self.d7 = tcnn_block(gf_dim*1*2,c_dim,4,2,1, first_layer = True)#1 x 128^3
   self.tanh = nn.Tanh()

 def forward(self,x):
   e1 = self.e1(x)
   #print(e1.shape,"e1")
   e2 = self.e2(F.leaky_relu(e1,0.2))
   #print(e2.shape,"e2")
   e3 = self.e3(F.leaky_relu(e2,0.2))
   #print(e3.shape,"e3")
   e4 = self.e4(F.leaky_relu(e3,0.2))
   #print(e4.shape,"e4")
   e5 = self.e5(F.leaky_relu(e4,0.2))
   #print(e5.shape,"e5")
   e6 = self.e6(F.leaky_relu(e5,0.2))
   #print(e6.shape,"e6")
   e7 = self.e7(F.leaky_relu(e6,0.2))
   #print(e7.shape,"e7")
   #e7 = self.e7(F.leaky_relu(e6,0.2))
   #e8 = self.e8(F.leaky_relu(e7,0.2))
   #d0=self.d0(e6)
   #print(d0.shape,"d0")
   d1 = torch.cat([F.dropout(self.d1(F.relu(e7)),0.5,training=True),e6],1)
   #print(d1.shape,"d1")
   d2 = torch.cat([F.dropout(self.d2(F.relu(d1)),0.5,training=True),e5],1)
   #print(d2.shape,"d2")
   d3 = torch.cat([F.dropout(self.d3(F.relu(d2)),0.5,training=True),e4],1)
   #print(d3.shape,"d3")
   d4 = torch.cat([self.d4(F.relu(d3)),e3],1)
   #print(d4.shape,"d4")
   d5 = torch.cat([self.d5(F.relu(d4)),e2],1)
   #print(d5.shape,"d5")
   d6 = torch.cat([self.d6(F.relu(d5)),e1],1)
   #print(d6.shape,"d6")
   d7 = self.d7(F.relu(d6))
   #print(d7.shape,"d7")


   return self.tanh(d7)



def test():
  x=torch.randn((1,1,128,128,128))
  model=Generator()
  preds=model(x)
  print(preds.shape)

if __name__ == '__main__':
    test()
