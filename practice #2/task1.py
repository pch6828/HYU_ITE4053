import sys
import time
import random
import numpy as np

def cross_entropy_loss(y_, y):
    return -(y*np.log(y_+1e-10)+((1-y)*np.log(1-y_+1e-10)))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def model(w, b, x):
    return sigmoid(np.dot(w, x)+b)

def compare(y_, y):
    return (y == 1 and y_ >= 0.5) or (y == 0 and y_ < 0.5)

def train_and_test(train_x, train_y, test_x, test_y, iteration, alpha, w, b, log_step):
    train_size = len(train_x)
    test_size = len(test_x)
    train_x = np.array(train_x).T
    train_y = np.array(train_y)
    test_x = np.array(test_x).T
    test_y = np.array(test_y)

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
        if log_step and (step+1) % 10 == 0:
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
    y_ = model(w, b, test_x)
    y_[y_>=0.5] = 1
    y_[y_<0.5] = 0
    test_accuracy = np.sum(y_==test_y)/test_size*100
    end = time.time()
    test_time = end-start
    return w, b, training_time, test_time, train_accuracy, test_accuracy

def main(argv):
    m = int(argv[1])
    n = int(argv[2])
    k = int(argv[3])
    r1 = random.uniform(-1,1)
    r2 = random.uniform(-1,1)
    r3 = random.uniform(-1,1)
    vectorwise_w = np.array([r1, r2])
    vectorwise_b = r3
    train_x, train_y, test_x, test_y = generate_dataset(m, n)
    
    vectorwise_result = train_and_test(train_x, train_y, test_x, test_y, k, 0.01, vectorwise_w, vectorwise_b, False)
    print('----------------RESULT----------------')
    print('Estimated W = ',vectorwise_result[0])
    print('Estimated B = ',vectorwise_result[1])
    print('Training Time = %f sec' % (vectorwise_result[2]))
    print('Testing Time  = %f sec' % (vectorwise_result[2]))
    print('Accuracy for Training Dataset = %.2f%%' % (vectorwise_result[4]))
    print('Accuracy for Testing Dataset  = %.2f%%' % (vectorwise_result[5]))

if __name__ == '__main__':
    main(sys.argv)
