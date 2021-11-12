# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:26:45 2021

  This is a template of a RNN - many to many for equal same sequential length
    Three RNN:
        -RNN
        -LSTM
        -GRU 

@author: Jorge Ivan Avalos Lopez
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import torch
import torch.nn as nn
from RNN_ManyToOne import RNN_MT1


class RNN_MTM(RNN_MT1):
    
    def __init__(self,input_dim,output_dim,hidden_dim, num_layers, 
                 batch_first = True, type_rnn = "RNN", **kwargs):
        super(RNN_MTM,self).__init__(input_dim,output_dim,hidden_dim, num_layers, 
                                     batch_first = True, type_rnn = "RNN", 
                                     **kwargs)
        
    def forward(self, input_features, *args):
        
        if self._type_rnn in ("RNN","GRU"):
            outputs, h_n = self.rnn(input_features,*args)
        else:
            outputs, (h_n,cn) = self.rnn(input_features,*args)
        
        # output of the h_n through the lineal 
        output = self.lineal(outputs)
            
        return output

if __name__ == "__main__":
    
    # Input tensor
    n_states = 5
    batch_size = 128
    input_dim = 4
    
    input_features = torch.randn(batch_size,n_states,input_dim)
    print(input_features.shape)
    
    # RNN- one-To-Many
    hidden_dim = 3
    output_dim = 3
    num_layers = 1
    batch_first = True    
    
    model_rnn = RNN_MTM(input_dim, output_dim, hidden_dim, num_layers, batch_first = True, type_rnn = "gru") 
    
    output = model_rnn(input_features)
    print(output.shape)