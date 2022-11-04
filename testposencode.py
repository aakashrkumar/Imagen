# import torch.nn as nn
import torch
import math
import numpy as np

# https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/ 
n = 10000
max_len = 1
d_model = 4
t=50
dim = 16
pe = np.zeros((1, d_model)) #torch.zeros(max_len, d_model)
print(pe.shape)
position = np.array([t]).reshape(-1, 1) #torch.arange(0, max_len).unsqueeze(1)
print(position.shape)
div_term = np.power(n, np.arange(0, d_model, 2) / d_model) #torch.pow(n, torch.arange(0, d_model, 2) / d_model)
print(div_term.shape)
pe[:, 0::2] = np.sin(position * div_term) #torch.sin(position * div_term)
print(pe.shape)
pe[:, 1::2] = np.cos(position * div_term) #torch.cos(position * div_term)
print(pe.shape)
print(pe)
pe = pe[np.newaxis,np.newaxis,:]
pe = np.repeat(pe, dim, axis=1)
pe = np.repeat(pe, dim, axis=2)
print(pe.shape)
print(pe)

# class PositionalEncoding(nn.Module):
#     "Implement the PE function."
#     def __init__(self, d_model, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
        
#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)], 
#                          requires_grad=False)
#         return self.dropout(x)