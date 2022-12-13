import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import tensorflow_datasets as tfds
import tqdm

import cv2

train_data, test_data = tfds.load(
    name="mnist",
    split=["train", "test"],
    as_supervised=True,
)

# dimensions of the input and latent spaces
input_dim = 784
latent_dim = 128

# size of the codebook
codebook_size = 1024

NUM_EPOCHS = 1000
BATCH_SIZE = 64

train_data = train_data.batch(BATCH_SIZE).shuffle(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE).shuffle(BATCH_SIZE)


class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(3 * 3 * 128, latent_dim)
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x, kernel_size=2)
    x = x.view(-1, 3 * 3 * 128)
    x = self.fc1(x)
    return x


class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(latent_dim, 3 * 3 * 128)
    self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
    self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
    self.conv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
  
  def forward(self, x):
    x = self.fc1(x)
    x = x.view(-1, 128, 3, 3)
    x = F.relu(self.conv1(x))
    x = F.interpolate(x, scale_factor=2)
    x = F.relu(   self.conv2(x))
    x = F.interpolate(x, scale_factor=2)
    x = F.relu(self.conv3(x))
    x = F.interpolate(x, scale_factor=2)
    return x

class VQVAE(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.codebook = nn.Parameter(torch.randn(codebook_size, latent_dim))

  
  def forward(self, x):
    # encode the input
    z = self.encoder(x)
    
    # find the nearest entry in the codebook
    z_q = self.quantize(z)
    
    # decode the quantized representation
    x_hat = self.decoder(z_q)
    
    return x_hat
  
  def quantize(self, z):
    # compute distances between each element of z and each entry in the codebook
    distances = torch.norm(z[:, None, :] - self.codebook, dim=2)
    
    # find the indices of the entries with the smallest distances
    _, indices = distances.min(dim=1)
    
    # select the entries with the smallest distances
    z_q = self.codebook[indices]
    
    return z_q

# create the VQ-VAE model
# load the model if it exists

model = VQVAE()
model.train()

# define the reconstruction loss
reconstruction_loss = nn.MSELoss()

# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# define the VQ-SGD optimizer
#vq_optimizer = optim.SGD(model.codebook, lr=1e-3, momentum=0.25)

# train the model
for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
  # loop over the training data
    losses = [] 
    for x in train_data:
        optimizer.zero_grad()
        # vq_optimizer.zero_grad()
        x = torch.tensor(x[0].numpy(), dtype=torch.float32)
        # flatten the input images
        x = x.view(-1, 1, 28, 28)

        # compute the reconstruction loss for the input
        loss = reconstruction_loss(x, model(x))
        losses.append(loss.item())
        # backpropagate and update the model parameters
        loss.backward()
        optimizer.step()
        # vq_optimizer.step()
    with torch.no_grad():
        for x in test_data:
            x = torch.tensor(x[0].numpy(), dtype=torch.float32)
            # flatten the input images
            # show 8 by 8 grid of original images
            original = x.view(-1, 28, 28).numpy()
            original = original.reshape(8, 8, 28, 28).transpose(0, 2, 1, 3).reshape(8*28, 8*28)
            cv2.imshow("Original", original)
            x = x.view(-1, 1, 28, 28)
            # compute the reconstruction loss for the input
            # backpropagate and update the model parameters
            # show the reconstructed images 8 by 8 grid
            reconstructed = model(x).view(-1, 28, 28).detach().numpy()
            reconstructed = reconstructed.reshape(8, 8, 28, 28).transpose(0, 2, 1, 3).reshape(8*28, 8*28)
            cv2.imshow("Reconstructed", reconstructed)
            cv2.waitKey(1)
            break
        torch.save(model.state_dict(), "model.pth")
    # print the average loss for the epoch
    print(f"epoch {epoch}: loss = {np.mean(losses):.3f}")



