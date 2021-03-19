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

def elementwise_model(w, b, x):
    return sigmoid(w[0]*x[0]+w[1]*x[1]+b)

def vectorwise_model(w, b, x):
    return sigmoid(np.dot(w, x)+b)

def compare(y_, y):
    return (y == 1 and y_ >= 0.5) or (y == 0 and y_ < 0.5)

def elementwise_training(train_x, train_y, test_x, test_y, iteration, alpha, w, b):
    train_size = len(train_x)
    test_size = len(test_x)
    print('Training with Element-wise Implementation...')
    for step in range(iteration):
        dw1 = 0
        dw2 = 0
        db = 0
        train_cost = 0
        test_cost = 0
        train_accuracy = 0
        test_accuracy = 0
        for i in range(train_size):
            x = train_x[i]
            y = train_y[i]

            y_ = elementwise_model(w, b, x)
            train_cost -= cross_entropy_loss(y_, y)

            if compare(y_, y):
                train_accuracy += 1

            dw1 += x[0]*(y_-y)
            dw2 += x[1]*(y_-y)
            db += (y_-y)
        
        for i in range(test_size):
            x = test_x[i]
            y = test_y[i]

            y_ = elementwise_model(w, b, x)
            test_cost -= cross_entropy_loss(y_, y)
            
            if compare(y_, y):
                test_accuracy += 1

        train_cost /= train_size
        test_cost /= test_size
        train_accuracy = train_accuracy / train_size * 100
        test_accuracy = test_accuracy / test_size * 100
        dw1 /= train_size
        dw2 /= train_size
        db /= train_size

        w[0] -= alpha*dw1
        w[1] -= alpha*dw2
        b -= alpha*db
        if (step+1) % 10 == 0:
            print('Iteration #',(step+1))
            print('Now W = ', w)
            print('Now B = ', b)
            print()
            print('Cost for Training Dataset = ', train_cost)
            print('Cost for Testing Dataset  = ', train_cost)
            print()
            print('Accuracy for Training Dataset = %.2f%%' % (train_accuracy))
            print('Accuracy for Testing Dataset  = %.2f%%' % (test_accuracy))
            print()

    return w, b

def vectorwise_training(train_x, train_y, test_x, test_y, iteration, alpha, w, b):
    train_size = len(train_x)
    test_size = len(test_x)
    train_x = np.array(train_x).T
    train_y = np.array(train_y)
    test_x = np.array(test_x).T
    test_y = np.array(test_y)

    print('Training with Vector-wise Implementation...')
    for step in range(iteration):
        train_cost = 0
        test_cost = 0
        train_accuracy = 0
        test_accuracy = 0
        
        y_ = vectorwise_model(w, b, train_x)
        train_cost = -cross_entropy_loss(y_, train_y).sum()/train_size

        dw = np.dot(train_x, y_ - train_y)/train_size
        db = (y_-train_y).sum()/train_size

        y_[y_>=0.5] = 1
        y_[y_<0.5] = 0
        train_accuracy = np.sum(y_==train_y)/train_size*100
        
        y_ = vectorwise_model(w, b, test_x)
        test_cost = -cross_entropy_loss(y_, test_y).sum()/test_size
        
        y_[y_>=0.5] = 1
        y_[y_<0.5] = 0
        test_accuracy = np.sum(y_==test_y)/test_size*100

        w = w - alpha*dw
        b -= alpha*db
        if (step+1) % 10 == 0:
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

    return w, b

def main(argv):
    m = int(argv[1])
    n = int(argv[2])
    k = int(argv[3])
    r1 = random.uniform(-1,1)
    r2 = random.uniform(-1,1)
    r3 = random.uniform(-1,1)
    elementwise_w = np.array([r1, r2])
    vectorwise_w = np.array([r1, r2])
    b = r3
    train_x, train_y, test_x, test_y = generate_dataset(m, n)
    
    elementwise_training(train_x, train_y, test_x, test_y, k, 0.01, elementwise_w, b)
    vectorwise_training(train_x, train_y, test_x, test_y, k, 0.01, vectorwise_w, b)

if __name__ == '__main__':
    main(sys.argv)
