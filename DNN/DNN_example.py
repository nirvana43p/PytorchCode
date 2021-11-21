# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:34:27 2021
    
    This an example of a simple DNN based on DNNTemplate.txt

@author: Jorge Ivan Avalos Lopez
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self,input_dim,dict_arch):
        super(DNN,self).__init__()
        
        self._input_dim = input_dim
        self._dict_arch = dict_arch
        
        self.layer1 = nn.Sequential(
                        nn.Linear(self._input_dim,self._dict_arch["layer1"]["input_dim"]),
                        nn.ReLU(),
                        nn.Dropout(p=self._dict_arch["layer1"]["P"])
                    )
        
        self.layer2 = nn.Sequential(
                        nn.Linear(self._dict_arch["layer1"]["input_dim"],self._dict_arch["layer2"]["input_dim"]),
                        nn.ReLU(),
                        nn.Dropout(p=self._dict_arch["layer1"]["P"])
                    )
        self.layer3 = nn.Sequential(
                        nn.Linear(self._dict_arch["layer2"]["input_dim"],self._dict_arch["layer3"]["input_dim"]),
                    )
        
        self.dnn = nn.Sequential(self.layer1,self.layer2,self.layer3)

    def forward(self,t_input):
        return self.dnn(t_input)



if __name__ == "__main__":
    input_dim = 100 # dimension of input data
    batch_dim = 128
    t_input = torch.randn(batch_dim,input_dim)
    print(t_input.shape)
    dict_arch = {
            "layer1" : {"input_dim":50,"P":0.4},
            "layer2" : {"input_dim":25,"P":0.5},
            "layer3" : {"input_dim":7}
        }
    dnn = DNN(input_dim,dict_arch)
    o_tensor = dnn(t_input)
    print(o_tensor.shape)
    

