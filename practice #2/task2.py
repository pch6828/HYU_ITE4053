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

def compare(y_, y):
    return (y == 1 and y_ >= 0.5) or (y == 0 and y_ < 0.5)

def train_and_test(train_x, train_y, test_x, test_y, iteration, alpha, w1, b1, w2, b2, log_step):
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
        
        a1 = model(w1, b1, train_x)
        y_ = model(w2, b2, np.array([a1]))
        train_cost = -cross_entropy_loss(y_, train_y).sum()/train_size

        dz2 = y_ - train_y
        dw2 = np.dot(a1, dz2)/train_size
        db2 = dz2.sum()/train_size
        
        dz1 = np.dot(w2.T, np.array([dz2]))*(a1*(1-a1))
        dw1 = np.dot(train_x, dz1)/train_size
        db1 = dz1.sum()/train_size

        y_[y_>=0.5] = 1
        y_[y_<0.5] = 0
        train_accuracy = np.sum(y_==train_y)/train_size*100
        
        if log_step:
            a1 = model(w1, b1, test_x)
            y_ = model(w2, b2, np.array([a1]))
            test_cost = -cross_entropy_loss(y_, test_y).sum()/test_size
            
            y_[y_>=0.5] = 1
            y_[y_<0.5] = 0
            test_accuracy = np.sum(y_==test_y)/test_size*100

        w1 = w1 - alpha*dw1
        b1 -= alpha*db1
        w2 = w2 - alpha*dw2
        b2 -= alpha*db2
        
        if log_step and (step+1) % 50 == 0:
            print('Iteration #',(step+1))
            print('Now W1 = ', w1)
            print('Now B1 = ', b1)            
            print('Now W2 = ', w2)
            print('Now B2 = ', b2)
            print()
            print('Cost for Training Dataset = ', train_cost)
            print('Cost for Testing Dataset  = ', test_cost)
            print()
            print('Accuracy for Training Dataset = %.2f%%' % (train_accuracy))
            print('Accuracy for Testing Dataset  = %.2f%%' % (test_accuracy))
            print()
    end = time.time()

    training_time = end-start
    a1 = model(w1, b1, train_x)
    y_ = model(w2, b2, np.array([a1]))
    y_[y_>=0.5] = 1
    y_[y_<0.5] = 0
    train_accuracy = np.sum(y_==train_y)/train_size*100
    
    start = time.time()
    y_ = []
    test_x = test_x.T
    for x in test_x:
        y_.append(model(w2, b2, np.array([model(w1, b1, x)])))
        
    y_ = np.array(y_)
    y_[y_>=0.5] = 1
    y_[y_<0.5] = 0
    test_accuracy = np.sum(y_==test_y)/test_size*100
    end = time.time()
    test_time = end-start
    
    return w1, b1, w2, b2, training_time, test_time, train_accuracy, test_accuracy

def main():
    m = 10000
    n = 500
    k = 5000

    train_filename = 'train_2018008395.npz'
    test_filename = 'test_2018008395.npz'
    
    r1 = random.uniform(-1,1)
    r2 = random.uniform(-1,1)
    r3 = random.uniform(-1,1)
    r4 = random.uniform(-1,1)
    r5 = random.uniform(-1,1)
    
    w1 = np.array([r1, r2])
    b1 = r3
    w2 = np.array([r4])
    b2 = r5

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
    
    result = train_and_test(train_x, train_y, test_x, test_y, k, 10, w1, b1, w2, b2, False)
    print('----------------RESULT----------------')
    print('Estimated W1 = ',result[0])
    print('Estimated B1 = ',result[1])     
    print('Estimated W2 = ',result[2])
    print('Estimated B2 = ',result[3])
    print('Training Time = %f sec' % (result[4]))
    print('Testing Time  = %f sec' % (result[5]))
    print('Accuracy for Training Dataset = %.2f%%' % (result[6]))
    print('Accuracy for Testing Dataset  = %.2f%%' % (result[7]))

if __name__ == '__main__':
    main()
