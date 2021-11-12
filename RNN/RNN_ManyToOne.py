# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:58:20 2021
    
    This is a template of a RNN - Many to one
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


class RNN_MT1(nn.Module):
    
    """ __init__ method creation

        Args:
            input_dim (Integer): Dimension of the input 
            hidden_dim (Integer): Dimension of the hidden state
            num_rnn (Integer): Number of stacked rnn
            batch_first (Bool): if the dim=0 goes the batch size
            type_rnn (String): 3 RNNs -> (RNN, LST and GRU)
            **kwargs (Dict): extra arguments to the RNNs 
    """
    
    def __init__(self, input_dim,output_dim, hidden_dim, num_layers, 
                 batch_first = True, type_rnn = "RNN", **kwargs):
        super(RNN_MT1,self).__init__()
        
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._batch_first = batch_first
        self._type_rnn = type_rnn.upper()
        
        
        if self._type_rnn == "RNN":
            self.rnn = nn.RNN(self._input_dim, self._hidden_dim, self._num_layers,batch_first=self._batch_first,**kwargs)
        elif self._type_rnn == "LSTM":
            self.rnn = nn.LSTM(self._input_dim, self._hidden_dim, self._num_layers,batch_first=self._batch_first,**kwargs)
        elif self._type_rnn == "GRU":
            self.rnn = nn.GRU(self._input_dim, self._hidden_dim, self._num_layers,batch_first=self._batch_first, **kwargs)
        else:
            raise Exception("{} is not defined".format(self._type_rnn))
            
        
        self.lineal = nn.Linear(self._hidden_dim,self._output_dim)
            
        
        """ forward method
            
            Args:
                input_features (torch.Tensor): input of the RNN, 
                                                if batch_first is True 
                                                    shape = (batch_size,n_states,input_dim)
                                                else 
                                                    shape = (n_states,batch_size,input_dim)
            Return:
                output (torch.Tensor): shape -> (1 if bidirectional is False, batch_size, output_dim)
                                                    
        """
    def forward(self, input_features, *args):
        
        # input through the RNN
        """
        output -> shape = (batch_size is True,n_states,hidden_dim)
        h_n -> shape = (1 if bidirectional is True, batch_size,hidden_dim)
        """
        if self._type_rnn in ("RNN","GRU"):
            
            outputs, h_n = self.rnn(input_features,*args)
        else:
            outputs, (h_n,cn) = self.rnn(input_features,*args)
            
        # output of the h_n through the lineal 
        output = self.lineal(h_n)
        
        return output



# Test
if __name__ == "__main__":
    
    # Input tensor
    n_states = 5
    batch_size = 128
    input_dim = 4
    
    input_features = torch.randn(batch_size,n_states,input_dim)
    print(input_features.shape)
    
    # RNN-Many-to-one
    output_dim = 1
    hidden_dim = 3
    num_layers = 1
    batch_first = True    
    
    model_rnn = RNN_MT1(input_dim,output_dim, hidden_dim, num_layers, batch_first = True, type_rnn = "gru") 
    
    output = model_rnn(input_features)
    print(output.shape)
        
        
        

