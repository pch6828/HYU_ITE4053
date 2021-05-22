import os
import time
import random
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, default=10000, help='# of train samples')
parser.add_argument('--n', type=int, default=500, help='# of test samples')
parser.add_argument('--k', type=int, default=5000, help='# of epochs')
parser.add_argument('--batch', type=int, default=10000, help='batch size')
parser.add_argument('--loss', type=str, default='BCE', help='loss function')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')

LOSS = {'BCE':tf.keras.losses.BinaryCrossentropy(),
        'MSE':tf.keras.losses.MeanSquaredError()}
OPTIMIZER = {'SGD':tf.keras.optimizers.SGD(learning_rate=0.01),
             'RMSProp':tf.keras.optimizers.RMSprop(learning_rate=0.01), 
             'Adam':tf.keras.optimizers.Adam(learning_rate=0.01)}

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

def main(m, n, k, batch, loss, opt):
    train_filename = 'train_2018008395.npz'
    test_filename = 'test_2018008395.npz'

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

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, input_shape=(2,), activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    
    model.compile(optimizer=OPTIMIZER[opt],
                  loss=LOSS[loss],
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    train_start = time.time()
    train_result = model.fit(x=train_x, 
                             y=train_y,
                             batch_size=batch,
                             epochs=k,
                             verbose=False)
    train_time = time.time() - train_start
    
    test_start = time.time()
    test_result = model.evaluate(x=test_x, 
                                 y=test_y, 
                                 verbose=False)
    
    test_time = time.time() - test_start

    print('----------------RESULT----------------')
    print('Accuracy for Training Dataset = %.2f%%' % (train_result.history['binary_accuracy'][-1]*100))
    print('Accuracy for Testing Dataset  = %.2f%%' % (test_result[1]*100))
    print('Training Time = %f sec' % train_time)
    print('Testing Time  = %f sec' % test_time)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.m, args.n, args.k, args.batch, args.loss, args.optimizer)