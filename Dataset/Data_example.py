# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:35:19 2021

    This is an example of a Dataset based on DataTemplate.txt.
    Iris plant dataset was used. 
    
@author: Jorge Ivan Avalos Lopez
python: 3.8.3
pytorch: 1.6.0
sklearn: 0.23.1
"""


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


class myData(Dataset):
    
    
    def __init__(self,path,transform = None, train = True, 
                 split_data = {"test_size" : 0.2, "random_state" : None}):
        super(myData, self).__init__()
        
        self._path = path
        self._transform = transform
        self._train = train
        self._split_data = split_data
        
        # Read the dataset from a csv, h5, json, html, etc.
        iris = datasets.load_iris()
        
        
        # Split X_data (input vector - feature vector) and Y_data(output_vector - label vector)
        # from de dataset
        self._X = iris.data
        self._Y = iris.target
        
        # Do some preprocessing if need it.
        # TODO ------------
        
        # Split dataset into train and test using train_test_split
        self._X_train, self._X_test, self._Y_train, self._Y_test = train_test_split(self._X,self._Y, 
                                                                                    test_size = self._split_data["test_size"],
                                                                                    random_state = self._split_data["random_state"])
        
    
        # Get the cardinality of the dataset
        if self._train:
            self._n_samples = len(self._X_train)
        else:
            self._n_samples = len(self._X_test)
        
    
    """ __getitem__ magic method to index the object
        
        Args:
            index (Integer) : Define the index
            
        Return:
            sample (Tuple) : (input vector, label vector)
    
    """
    def __getitem__(self, index):
        
        if self._train:
            sample = self._X_train[index], self._Y_train[index]
        else:
            sample = self._X_test[index], self._Y_test[index]
        
        if self._transform:
            sample = self._transform(sample)
        
        return sample
                
                
        
    """ __len__ magic method to len the object
    
    """
    def __len__(self):
        return self._n_samples
        
        
class ToTensor:
    
    """ __call__ magic method to recive objects and transform them 
        
        Return: 
            (torch.Tensor, torch.Tensor)
    """
    def __call__(self, sample):
        x, y = sample
        return torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(y))



# Test
if __name__ == "__main__":
    data_train = myData(None,transform=ToTensor())
    data_test = myData(None,transform=ToTensor(), train = False)
    
    
    trainloader = DataLoader(dataset=data_train,batch_size= 16,shuffle=True)
    testloader = DataLoader(dataset=data_test,batch_size= 16,shuffle=False)
    
    
    # Run the batches
    for x_train, y_train in trainloader:
        print(x_train.shape, y_train.shape, type(x_train),type(y_train))

    print()
    # Run the batches
    for x_test, y_test in testloader:
        print(x_test.shape, y_test.shape, type(x_test.shape),type(y_test))

    