import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import os
from gen import Generator
from disc import Discriminator
from dataset_loader import train_loader, val_loader
import logging
#from bigdata import Data_Generator
torch.backends.cudnn.benchmark = True

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

def train_fn(D, G, train_loader, val_loader, D_optimizer, G_optimizer, num_eps, device):
  epochs=num_eps
  D=D
  G=G
  train_loader=train_loader
  val_loader=val_loader
  D_optimizer=D_optimizer
  G_optimizer=G_optimizer
  device=device


  L1_lambda = 100.0
  in_w = in_h = in_d = 64 #dimensions of a single data sample



  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()
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
  os.makedirs('PlotsJuly4', exist_ok=True)
  save_dir1='PlotsJuly4/' #where to save plots
  os.makedirs('ModelsJuly4', exist_ok=True)
  save_dir='ModelsJuly4/' #where to save models
  bce_criterion = nn.BCEWithLogitsLoss() #nn.BCELoss()
  L1_criterion = nn.L1Loss()
  logging.info('starting training')
  for ep in range(epochs):
    for i, data in enumerate(train_loader):
      #for batch_input, batch_output in data_generator.generate_train_batch():
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
      with torch.cuda.amp.autocast():
        real_patch = D(x,y)
        real_gan_loss=bce_criterion(real_patch,real_class)
        D_loss_real_tr_iter.append(real_gan_loss.item())   #TRAINING loss, D on real, for each iteration
        fake=G(x)
        fake_patch = D(x,fake.detach())
        fake_gan_loss=bce_criterion(fake_patch,fake_class)
        D_loss_fake_tr_iter.append(fake_gan_loss.item())  #training loss, D on fake, every iteration
        D_loss = (real_gan_loss + fake_gan_loss)/2 #can not also divide by 2

      # D_loss.backward()
      # D_optimizer.step()
      #mixed precision training
      d_scaler.scale(D_loss).backward()
      d_scaler.step(D_optimizer)
      d_scaler.update()

      #Train G
      G.zero_grad()
      with torch.cuda.amp.autocast():
        fake_patch = D(x,fake)
        fake_gan_loss=bce_criterion(fake_patch,real_class)
        L1_loss = L1_criterion(fake,y)
        G_loss = fake_gan_loss + L1_lambda*L1_loss
        G_loss_tr_iter.append(G_loss.item())         #training loss, G total, every iteration

      # G_loss.backward()
      # G_optimizer.step()
      g_scaler.scale(G_loss).backward()
      g_scaler.step(G_optimizer)
      g_scaler.update()

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
      with torch.no_grad(), torch.cuda.amp.autocast():
        real_patch = D(x,y)
        real_gan_loss=bce_criterion(real_patch,real_class)
        D_loss_real_val_iter.append(real_gan_loss.item())   #Val loss, D on real, for each iteration
        fake=G(x)
        fake_patch = D(x,fake.detach())
        fake_gan_loss=bce_criterion(fake_patch,fake_class)
        D_loss_fake_val_iter.append(fake_gan_loss.item())  #Val loss, D on fake, every iteration
        D_loss = (real_gan_loss + fake_gan_loss)/2 #can not also divide by 2

      #Val G
      with torch.no_grad(), torch.cuda.amp.autocast():
        fake_patch = D(x,fake)
        fake_gan_loss=bce_criterion(fake_patch,real_class)
        L1_loss = L1_criterion(fake,y)
        G_loss = fake_gan_loss + L1_lambda*L1_loss
        G_loss_val_iter.append(G_loss.item())         #Val loss, G total, every iteration

    #Validation complete for the epoch
    #saving situation
    if np.mean(G_loss_val_iter) < G_loss_val_min:
      print(f"validation loss reduced at epoch {ep}")
      logging.info(f'validation loss reduced at epoch {ep}')
      #save best model based on val loss
      torch.save(G.state_dict(), save_dir+'Vbestmodel_Gen.pth')
      torch.save(G_optimizer.state_dict(),save_dir+'Vbestmodel_GenO.pth')
      torch.save(D.state_dict(), save_dir+'Vbestmodel_Disc.pth')
      torch.save(D_optimizer.state_dict(),save_dir+'Vbestmodel_DiscO.pth')
      G_loss_val_min=np.mean(G_loss_val_iter)

    # if ep%20==0:
    #   torch.save(G.state_dict(), save_dir+f'model_Gen_{epoch}.pth')

    #Validation losses for the epoch
    G_loss_val_epoch.append(np.mean(G_loss_val_iter))
    G_loss_val_iter=[]
    D_loss_real_val_epoch.append(np.mean(D_loss_real_val_iter))
    D_loss_real_val_iter=[]
    D_loss_fake_val_epoch.append(np.mean(D_loss_fake_val_iter))
    D_loss_fake_val_iter=[]
    logging.info(f'training success for this epoch and losses loaded for epoch {ep}')

  #training is complete, save the losses for plotting
  np.save(save_dir1+'val_loss_G.npy',G_loss_val_epoch)
  np.save(save_dir1+'val_loss_D_fake.npy',D_loss_fake_val_epoch)
  np.save(save_dir1+'val_loss_D_real.npy',D_loss_real_val_epoch)
  np.save(save_dir1+'train_loss_G.npy',G_loss_tr_epoch)
  np.save(save_dir1+'train_loss_D_fake.npy',D_loss_fake_tr_epoch)
  np.save(save_dir1+'train_loss_D_real.npy',D_loss_real_tr_epoch)
  #save last model
  torch.save(G.state_dict(), save_dir+'lastmodel_Gen.pth')
  torch.save(G_optimizer.state_dict(),save_dir+'lastmodel_GenO.pth')
  torch.save(D.state_dict(), save_dir+'lastmodel_Disc.pth')
  torch.save(D_optimizer.state_dict(),save_dir+'lastmodel_DiscO.pth')
  logging.info(f'saved last model at epoch{ep}')

def main():
  torch.manual_seed(545667)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  gf_dim = 64
  df_dim = 64
  c_dim = 1 #color channels, 1 for grayscale
  G = Generator().to(device)
  D = Discriminator().to(device)
  G_optimizer = optim.Adam(G.parameters(), lr=1e-5,betas=(0.5,0.999))
  D_optimizer = optim.Adam(D.parameters(), lr=1e-5,betas=(0.5,0.999))
  logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',filename='Oct06train.log', level=logging.DEBUG)
  num_eps=2
  G.apply(initialize_weights)
  D.apply(initialize_weights)
  #enable to continue training
  # load_dir=
  # G.load_state_dict(torch.load(load_dir+'lastmodel_Gen.pth'))
  # G_optimizer.load_state_dict(torch.load(load_dir+'lastmodel_GenO.pth'))
  # D.load_state_dict(torch.load(load_dir+'lastmodel_Disc.pth'))
  # D_optimizer.load_state_dict(torch.load(load_dir+'lastmodel_DiscO.pth'))

  train_fn(D, G, train_loader, val_loader, D_optimizer, G_optimizer, num_eps, device)
  logging.info('mission successful')

if __name__ == "__main__":
  main()

