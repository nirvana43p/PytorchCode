# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:58:20 2021
    
    This is a template of a RNN - Many to one

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
    
    def __init__(self, input_dim, hidden_dim, num_layers, 
                 batch_first = True, type_rnn = "RNN", **kwargs):
        super(RNN_MT1,self).__init__()
        
        self._input_dim = input_dim
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
            
            
        
        """ forward method
            
            Args:
                input_features (torch.Tensor): input of the RNN, 
                                                if batch_first is True 
                                                    shape = (batch_size,n_states,input_dim)
                                                else 
                                                    shape = (n_states,batch_size,input_dim)
            Return:
                output (tuple((torch.Tensor),(torch.Tensor))),: output of the RNN
                                        if batch_first is True 
                                                    output -> shape = (batch_size,n_states,hidden_dim)
                                                    h_f -> shape = (1, batch_size,hidden_dim)
                                                else 
                                                    output -> shape = (n_states,batch_size,hidden_state)
                                        h_f -> shape = (1, batch_size,hidden_dim)
                                                    
        """
    def forward(self, input_features, *args):
        
        if self._type_rnn in ("RNN","GRU"):
            output, h_n = self.rnn(input_features,*args)
            return output, h_n
        else:
            output, (h_n,cn) = self.rnn(input_features,*args)
            return output, (h_n,cn)



# Test
if __name__ == "__main__":
    
    # Input tensor
    n_states = 5
    batch_size = 128
    input_dim = 4
    
    input_features = torch.randn(batch_size,n_states,input_dim)
    
    # RNN 
    hidden_dim = 3
    num_layers = 1
    batch_first = True    
    
    model_rnn = RNN_MT1(input_dim, hidden_dim, num_layers, batch_first = True, type_rnn = "gru") 
    
    output = model_rnn(input_features)
        
        
        

