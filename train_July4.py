import numpy as np
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',filename='July5train_B.log', level=logging.DEBUG)

#train setup
G = Generator().to(device)
D = Discriminator().to(device)
G_optimizer = optim.Adam(G.parameters(), lr=1e-5,betas=(0.5,0.999))
D_optimizer = optim.Adam(D.parameters(), lr=1e-5,betas=(0.5,0.999))

bce_criterion = nn.BCEWithLogitsLoss() #nn.BCELoss()
L1_criterion = nn.L1Loss()


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

#verbose=True
G.apply(initialize_weights)
D.apply(initialize_weights)

G_loss_tr_epoch=[]
G_loss_tr_iter=[]
D_loss_real_tr_epoch=[]
D_loss_real_tr_iter=[]
D_loss_fake_tr_epoch=[]
D_loss_fake_tr_iter=[]

G_loss_val_epoch=[]
G_loss_val_iter=[]
D_loss_real_val_epoch=[]
D_loss_real_val_iter=[]
D_loss_fake_val_epoch=[]
D_loss_fake_val_iter=[]

G_loss_val_min=100
logging.info('starting training')

for ep in range(epochs):
 for i, data in enumerate(train_loader):
  G.train()
  D.train()
  x, y = data
  x=x.unsqueeze(1)
  y=y.unsqueeze(1)
  #print(x.size())
  #print(y.size())
  x = x.to(device)
  y = y.to(device)

  b_size = x.shape[0]

  real_class = torch.ones(b_size,1,14,14,14).to(device)
  fake_class = torch.zeros(b_size,1,14,14,14).to(device)

  #Train D
  D.zero_grad()

  real_patch = D(x,y)
  real_gan_loss=bce_criterion(real_patch,real_class)
  D_loss_real_tr_iter.append(real_gan_loss.item())   #TRAINING loss, D on real, for each iteration

  fake=G(x)

  fake_patch = D(x,fake.detach())
  fake_gan_loss=bce_criterion(fake_patch,fake_class)
  D_loss_fake_tr_iter.append(fake_gan_loss.item())  #training loss, D on fake, every iteration

  D_loss = (real_gan_loss + fake_gan_loss)/2 #can not also divide by 2
  D_loss.backward()
  D_optimizer.step()


  #Train G
  G.zero_grad()
  fake_patch = D(x,fake)
  fake_gan_loss=bce_criterion(fake_patch,real_class)

  L1_loss = L1_criterion(fake,y)
  G_loss = fake_gan_loss + L1_lambda*L1_loss
  G_loss_tr_iter.append(G_loss.item())         #training loss, G total, every iteration
  G_loss.backward()
  G_optimizer.step()
  
 #complete training for one whole set of train loader
 #use of np.mean is applicable ONLY if batch size is 1. Else, change it to sum/num of iterations. Make sure data loader length is % by bs
 G_loss_tr_epoch.append(np.mean(G_loss_tr_iter))
 G_loss_tr_iter=[]
 D_loss_real_tr_epoch.append(np.mean(D_loss_real_tr_iter))
 D_loss_real_tr_iter=[]
 D_loss_fake_tr_epoch.append(np.mean(D_loss_fake_tr_iter))
 D_loss_fake_tr_iter=[]
 
 #validation for this epoch
 G.eval()
 D.eval()

 for i, data in enumerate(val_loader):
  x, y = data
  x=x.unsqueeze(1)
  y=y.unsqueeze(1)
  #print(x.size())
  #print(y.size())
  x = x.to(device)
  y = y.to(device)

  b_size = x.shape[0]

  real_class = torch.ones(b_size,1,14,14,14).to(device)
  fake_class = torch.zeros(b_size,1,14,14,14).to(device)

  #Val D
  real_patch = D(x,y)
  real_gan_loss=bce_criterion(real_patch,real_class)
  D_loss_real_val_iter.append(real_gan_loss.item())   #Val loss, D on real, for each iteration

  fake=G(x)

  fake_patch = D(x,fake.detach())
  fake_gan_loss=bce_criterion(fake_patch,fake_class)
  D_loss_fake_val_iter.append(fake_gan_loss.item())  #Val loss, D on fake, every iteration
  D_loss = (real_gan_loss + fake_gan_loss)/2 #can not also divide by 2
  D_loss.backward()
  D_optimizer.step()

  #Val G
  G.zero_grad()
  fake_patch = D(x,fake)
  fake_gan_loss=bce_criterion(fake_patch,real_class)

  L1_loss = L1_criterion(fake,y)
  G_loss = fake_gan_loss + L1_lambda*L1_loss
  G_loss_val_iter.append(G_loss.item())         #Val loss, G total, every iteration
  G_loss.backward()
  G_optimizer.step()

 #Validation complete for the epoch
 #saving situation
 if np.mean(G_loss_val_iter) < G_loss_val_min:
  print(f'validation loss reduced at epoch {ep}')
  logging.info(f'validation loss reduced at epoch {ep}')
  #save best model based on val loss
  os.makedirs('ModelsJuly5', exist_ok=True)
  torch.save(Generator.state_dict(G), 'ModelsJuly5/Vbestmodel_Gen.pth')
  torch.save(G_optimizer.state_dict(),'ModelsJuly5/Vbestmodel_GenO.pth')
  torch.save(Discriminator.state_dict(D), 'ModelsJuly5/Vbestmodel_Disc.pth')
  torch.save(D_optimizer.state_dict(),'ModelsJuly5/Vbestmodel_DiscO.pth')
  G_loss_val_min=np.mean(G_loss_val_iter)

 #Validation losses for the epoch
 G_loss_val_epoch.append(np.mean(G_loss_val_iter))
 G_loss_val_iter=[]
 D_loss_real_val_epoch.append(np.mean(D_loss_real_val_iter))
 D_loss_real_val_iter=[]
 D_loss_fake_val_epoch.append(np.mean(D_loss_fake_val_iter))
 D_loss_fake_val_iter=[]
 
 #training is complete for the epoch, save the losses for plotting
 logging.info(f'training success for this epoch and saving plots for epoch {ep}')
 print(f'training success for this epoch and saving plots for epoch {ep}')
 os.makedirs('PlotsJuly5', exist_ok=True)
 np.save('PlotsJuly5/val_loss_G.npy',G_loss_val_epoch)
 np.save('PlotsJuly5/val_loss_D_fake.npy',D_loss_fake_val_epoch)
 np.save('PlotsJuly5/val_loss_D_real.npy',D_loss_real_val_epoch)
 np.save('PlotsJuly5/train_loss_G.npy',G_loss_tr_epoch)
 np.save('PlotsJuly5/train_loss_D_fake.npy',D_loss_fake_tr_epoch)
 np.save('PlotsJuly5/train_loss_D_real.npy',D_loss_real_tr_epoch)


logging.info('mission successful')




