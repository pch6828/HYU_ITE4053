import sys
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

def output_process(x, y, filename):
    file = open(filename, mode='w')

    for i in range(len(y)):
        for xi_j in x[i]:
            file.write(str(xi_j))
            file.write(str('\t'))
        file.write(str(y[i]))
        file.write('\n')

    file.close()

def main(argv):
    m = int(argv[1])
    n = int(argv[2])
    train_filename = argv[3]
    test_filename = argv[4]
    train_x, train_y, test_x, test_y = generate_dataset(m, n)
    
    output_process(train_x, train_y, train_filename)
    output_process(test_x, test_y, test_filename)

if __name__ == '__main__':
    main(sys.argv)