# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:35:12 2018
@author: childrenbody
"""
from perceptron import Perceptron

Linear = lambda x: x
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, Linear)
        
if __name__ == '__main__':
    import numpy as np
    
    data = np.random.random([10, 3])
    label = np.array([a*3 + b*2 - c for a, b, c in data])
    lu = LinearUnit(3)
    lu.train(data, label, 100, 0.1)
    
    test = np.random.random([10, 3])
    v = np.array([a*3 + b*2 - c for a, b, c in test])
    pred = np.array([lu.predict(_)[0] for _ in test]).reshape(10,)
    for i in range(10):
        print('test : {}\npredict : {}'.format(test[i], pred[i]))
    print('error : {}'.format(np.sum(np.abs(v - pred))/np.sum(v)))
