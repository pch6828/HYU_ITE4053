import sys
import time
import random
import numpy as np

def generate_dataset(num_train, num_test):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for _ in range(num_train):
        x = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        train_x.append(x)
        if x.sum() > 0:
            train_y.append(1)
        else:
            train_y.append(0)

    for _ in range(num_test):
        x = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        test_x.append(x)
        if x.sum() > 0:
            test_y.append(1)
        else:
            test_y.append(0)

    return train_x, train_y, test_x, test_y

def cross_entropy_loss(y_, y):
    return -(y*np.log(y_+1e-10)+((1-y)*np.log(1-y_+1e-10)))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def model(w, b, x):
    return sigmoid(np.dot(w, x)+b)

def compare(y_, y):
    return (y == 1 and y_ >= 0.5) or (y == 0 and y_ < 0.5)

def training(train_x, train_y, test_x, test_y, iteration, alpha, w, b, log_step):
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

    running_time = end-start
    y_ = model(w, b, train_x)
    y_[y_>=0.5] = 1
    y_[y_<0.5] = 0
    train_accuracy = np.sum(y_==train_y)/train_size*100
    y_ = model(w, b, test_x)
    y_[y_>=0.5] = 1
    y_[y_<0.5] = 0
    test_accuracy = np.sum(y_==test_y)/test_size*100
    return w, b, running_time, train_accuracy, test_accuracy

def main(argv):
    mode = argv[1]
    m = int(argv[2])
    n = int(argv[3])
    k = int(argv[4])
    r1 = random.uniform(-1,1)
    r2 = random.uniform(-1,1)
    r3 = random.uniform(-1,1)
    vectorwise_w = np.array([r1, r2])
    vectorwise_b = r3
    train_x, train_y, test_x, test_y = generate_dataset(m, n)
    
    if mode == 'log_step':
        print('Training with Vector-wise Implementation...')
        vectorwise_result = training(train_x, train_y, test_x, test_y, k, 0.01, vectorwise_w, vectorwise_b, True)
        print('----------------RESULT----------------')
        print('Estimated W = ',vectorwise_result[0])
        print('Estimated B = ',vectorwise_result[1])
        print('Running Time = %f sec' % (vectorwise_result[2]))
        print('Accuracy for Training Dataset = %.2f%%' % (vectorwise_result[3]))
        print('Accuracy for Testing Dataset  = %.2f%%' % (vectorwise_result[4]))
    elif mode == 'alpha':
        alpha_range = 1
        optimized_range = None
        optimized_alpha = None
        optimized_accracy = 0
        print('Search the Best Alpha Range...')

        for i in range(5):
            alpha_range/=10
            vectorwise_w = np.array([r1, r2])
            vectorwise_b = r3
            vectorwise_result = training(train_x, train_y, test_x, test_y, k, alpha_range, vectorwise_w, vectorwise_b, False)
            if optimized_accracy < vectorwise_result[3]:
                optimized_accracy = vectorwise_result[3]
                optimized_range = alpha_range
                optimized_alpha = alpha_range
        
        print('Search the Best Alpha Value...')
        for i in range(1, 10):
            alpha = optimized_range*i
            vectorwise_w = np.array([r1, r2])
            vectorwise_b = r3
            vectorwise_result = training(train_x, train_y, test_x, test_y, k, alpha, vectorwise_w, vectorwise_b, False)
            if optimized_accracy < vectorwise_result[3]:
                optimized_accracy = vectorwise_result[3]
                optimized_alpha = alpha
        print('----------------RESULT----------------')
        print('Best Alpha : ', optimized_alpha)
    else:
        print('Training with Vector-wise Implementation...')
        vectorwise_result = training(train_x, train_y, test_x, test_y, k, 0.01, vectorwise_w, vectorwise_b, False)
        print('----------------RESULT----------------')
        if mode == 'compare_all':
            print('Estimated W = ',vectorwise_result[0])
            print('Estimated B = ',vectorwise_result[1])
            print('Running Time = %f sec' % (vectorwise_result[2]))
            print('Accuracy for Training Dataset = %.2f%%' % (vectorwise_result[3]))
            print('Accuracy for Testing Dataset  = %.2f%%' % (vectorwise_result[4]))
        elif mode == 'time':
            print('Running Time = %f sec' % (vectorwise_result[2]))
        elif mode == 'parameter':
            print('Estimated W = ',vectorwise_result[0])
            print('Estimated B = ',vectorwise_result[1])
        elif mode == 'accuracy':
            print('Accuracy for Training Dataset = %.2f%%' % (vectorwise_result[3]))
            print('Accuracy for Testing Dataset  = %.2f%%' % (vectorwise_result[4]))
if __name__ == '__main__':
    main(sys.argv)
