# Practice #2
# Task 1 : binary classification using logistic regression (cross-entropy loss)
# Requirement
#   1. python 3 (I used python 3.8.2)
#   2. numpy module (I used numpy 1.18.2)
# Usage
#   1. print each step : python task1.py step
#   2. get final result with zero valued w, b : python task1.py zero
#   3. get final result with random w, b : python task1.py
# This program use dataset files named "train_2018008395.npz" and "test_2018008395.npz"
# If files do not exist, program will generate random fileset and save it as files

import os
import sys
import time
import random
import numpy as np

def read_dataset(filename):
    data = np.load(filename)
    x = data['x']
    y = data['y']
    return x, y

def generate_and_save_dataset(filename, size):
    x = []
    y = []
    for _ in range(size):
        temp = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        x.append(temp)

        if temp.sum() > 0:
            y.append(1)
        else:
            y.append(0)
    
    x = np.array(x)
    y = np.array(y)

    np.savez(filename, x=x, y=y)

    return x, y

def cross_entropy_loss(y_, y):
    return -(y*np.log(y_+1e-10)+((1-y)*np.log(1-y_+1e-10)))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def model(w, b, x):
    return sigmoid(np.dot(w, x)+b)

def train_and_test(train_x, train_y, test_x, test_y, iteration, alpha, w, b, log_step):
    train_size = train_x.shape[0]
    test_size = test_x.shape[0]

    train_x = train_x.T
    test_x = test_x.T

    start = time.time()
    for step in range(iteration):
        train_cost = 0
        test_cost = 0
        train_accuracy = 0
        test_accuracy = 0
        
        y_ = model(w, b, train_x)
        train_cost = -cross_entropy_loss(y_, train_y).sum()/train_size

        dw = np.dot(train_x, y_ - train_y)/train_size
        db = (y_-train_y).sum()/train_size

        y_[y_>=0.5] = 1
        y_[y_<0.5] = 0
        train_accuracy = np.sum(y_==train_y)/train_size*100
        
        if log_step:
            y_ = model(w, b, test_x)
            test_cost = -cross_entropy_loss(y_, test_y).sum()/test_size
            
            y_[y_>=0.5] = 1
            y_[y_<0.5] = 0
            test_accuracy = np.sum(y_==test_y)/test_size*100

        w = w - alpha*dw
        b -= alpha*db
        if log_step and (step+1) % 50 == 0:
            print('Iteration #',(step+1))
            print('Now W = ', w)
            print('Now B = ', b)
            print()
            print('Cost for Training Dataset = ', train_cost)
            print('Cost for Testing Dataset  = ', test_cost)
            print()
            print('Accuracy for Training Dataset = %.2f%%' % (train_accuracy))
            print('Accuracy for Testing Dataset  = %.2f%%' % (test_accuracy))
            print()
    end = time.time()

    training_time = end-start
    y_ = model(w, b, train_x)
    y_[y_>=0.5] = 1
    y_[y_<0.5] = 0
    train_accuracy = np.sum(y_==train_y)/train_size*100
    
    start = time.time()
    y_ = []
    test_x = test_x.T
    for x in test_x:
        y_.append(model(w, b, x))
    y_ = np.array(y_)
    y_[y_>=0.5] = 1
    y_[y_<0.5] = 0
    test_accuracy = np.sum(y_==test_y)/test_size*100
    end = time.time()
    test_time = end-start
    
    return w, b, training_time, test_time, train_accuracy, test_accuracy

def main(argv):
    m = 10000
    n = 500
    k = 5000

    mode = None
    if len(argv)>1:
        mode = argv[1]

    train_filename = 'train_2018008395.npz'
    test_filename = 'test_2018008395.npz'
    
    r1 = random.uniform(-10,10)
    r2 = random.uniform(-10,10)
    r3 = random.uniform(-10,10)

    w = np.array([r1, r2])
    b = r3

    if mode == 'zero':
        w = np.array([0.0,0.0])
        b = 0.0

    train_x = None
    train_y = None
    test_x = None
    test_y = None

    if os.path.isfile(train_filename) and os.path.isfile(test_filename):
        train_x, train_y = read_dataset(train_filename)
        test_x, test_y = read_dataset(test_filename)
    else:
        train_x, train_y = generate_and_save_dataset(train_filename, m)
        test_x, test_y = generate_and_save_dataset(test_filename, n)
    
    result = train_and_test(train_x, train_y, test_x, test_y, k, 3, w, b, mode=='step')
    print('----------------RESULT----------------')
    print('Estimated W = ',result[0])
    print('Estimated B = ',result[1])
    print('Training Time = %f sec' % (result[2]))
    print('Testing Time  = %f sec' % (result[3]))
    print('Accuracy for Training Dataset = %.2f%%' % (result[4]))
    print('Accuracy for Testing Dataset  = %.2f%%' % (result[5]))

if __name__ == '__main__':
    main(sys.argv)
