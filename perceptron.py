#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:53:45 2018

@author: childrenbody
"""
import numpy as np

class Perceptron:
    "perceptron machine, input units and activator function"
    def __init__(self, input_num, activator):
        self.activator = activator
        self.bais = 0
        self.weight = np.zeros([input_num, 1])
        self.num = input_num
        
    def __str__(self):
        "print weight and bais"
        return 'weight : {} \nbias : {}'.format(self.weight, self.bais)
    
    def predict(self, input_vec):
        input_vec = np.array(input_vec).reshape(self.num, 1)
        return self.activator(np.dot(self.weight.T, input_vec) + self.bais)
    
    def train(self, input_vecs, labels, iteration, rate):
        input_vecs = np.array(input_vecs)
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
    
    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for input_vec, label in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, label, output, rate)
        
    def _update_weights(self, input_vec, label, output, rate):
        delta = label - output
        self.weight = self.weight + rate*delta*input_vec.reshape(self.num, 1)
        self.bais += rate*delta
        
step = lambda x: 1 if x > 0 else 0

if __name__ == '__main__':
    def make_test_data(num):
        data = np.random.choice([1, 0], [num, 2])
        label = np.array([a and b for a, b in data])
        return data, label
    
    train, label = make_test_data(10)
    p = Perceptron(2, step)
    p.train(train, label, 10, 0.1)
    print(p)
    x, y = make_test_data(10)
    predict = [p.predict(_) for _ in x]
    for i in range(10):
        print("{} : {}".format(x[i], predict[i]))
    z = [1 if predict[i] == y[i] else 0 for i in range(len(y))]
    print('accuracy: {}'.format(sum(z)/len(y)))
    
            
    
        
    
    
