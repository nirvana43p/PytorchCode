# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:58:05 2021

    This is a template for train a DNN model

@author: Jorge Ivan Avalos Lopez
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import math



def train_model(model, loss, optimizer, data_train, data_test,
                num_epochs = 10, batch_size = 128, device = 'cuda'):
    
    # Build The DataLoader Object to make batches in training
    trainloader = DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True)
    testloader = DataLoader(dataset=data_test,batch_size=batch_size,shuffle=False)
    
    # number of iterations per epoch
    n_iterations_train = math.ceil(len(trainloader))
    n_iterations_test = math.ceil(len(testloader))
    
    # to store errors
    train_err = []
    test_err = []
    
    for epoch in range(num_epochs):
        
        train_error = 0
        for i, (x_train,y_train) in enumerate(trainloader):
            x_train,y_train = x_train.to(device),y_train.to(device)
            optimizer.zero_grad()
            output = model(x_train)
            l = loss(output,y_train)
            l.backward()
            optimizer.step()
            train_error += l.item()
        
        train_error_avg = train_error/n_iterations_train
        print("Train ------> epoch : {0}/{1}, loss : {2}".format(epoch+1,num_epochs,train_error_avg))
        train_err.append(train_error_avg)

        
        with torch.no_grad():
            test_error = 0
            for i, (x_test, y_test) in enumerate(testloader):
                x_test, y_test = x_test.to(device), y_test.to(device)
                output = model.eval()(x_test)
                l = loss(output,y_test)
                test_error += l.item()
             
            test_error_avg = test_error/n_iterations_test
            print("Test ------> epoch : {0}/{1}, loss : {2}".format(epoch+1,num_epochs,test_error_avg))
            test_err.append(test_error_avg)
            
        print("-"*20)
    
    return model



