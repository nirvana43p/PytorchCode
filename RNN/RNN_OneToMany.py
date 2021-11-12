# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:26:45 2021

  This is a template of a RNN - one to many
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


class RNN_1TM(nn.Module):
    
    def __init__(self, input_dim, seq_length, hidden_dim, 
                 type_rnn = "RNN", **kwargs):
        super(RNN_1TM,self).__init__()
        self._input_dim = input_dim
        self._output_dim = input_dim
        self._seq_length = seq_length
        self._hidden_dim = hidden_dim
        self._type_rnn = type_rnn.upper()
        
                
        if self._type_rnn == "RNN":
            self.rnnCell = nn.RNNCell(self._input_dim, self._hidden_dim,**kwargs)
        elif self._type_rnn == "LSTM":
            self.rnnCell = nn.LSTMCell(self._input_dim, self._hidden_dim,**kwargs)
        elif self._type_rnn == "GRU":
            self.rnnCell = nn.GRUCell(self._input_dim, self._hidden_dim,**kwargs)
        else:
            raise Exception("{} is not defined".format(self._type_rnn))
        
        self.lineal = nn.Linear(self._hidden_dim, self._output_dim)
        
        
    def forward(self, input_features, h0, c0):
        output = input_features
        h_out = h0
        c_out = c0
        
        if self._type_rnn in ("RNN","GRU"):
            for seq in range(self._seq_length):
                h_out = self.rnnCell(output,h_out)
                output = self.lineal(h_out)
        else:
            for seq in range(self._seq_length):
                h_out, c_out = self.rnnCell(output,(h_out,c_out))
                output = self.lineal(h_out)
        
        return output
            

if __name__ == "__main__":
    
    # Input tensor
    seq_length = 5
    batch_size = 128
    input_dim = 5
    
    input_features = torch.randn(batch_size,input_dim)
    print(input_features.shape)
    
    # RNN- one-To-Many
    hidden_dim = 10
    seq_length = 5 # or n_states
    
    h0_random = torch.randn(batch_size,hidden_dim)
    c0_random = torch.randn(batch_size,hidden_dim)

    model_rnn = RNN_1TM(input_dim, seq_length, hidden_dim, type_rnn = "lstm") 
    
    output = model_rnn(input_features,h0_random,c0_random)
    print(output.shape)
    
