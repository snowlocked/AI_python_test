# -*-coding:utf8-*-

import random
import numpy as np

def build_data(n):
    xs = []
    ys = []
    for i in range(2000):
        k = random.uniform(1,50)

        # x[i] = sin(k+i) (i=0,1,2...,n-1)
        # y[i] = sin(k+n)
        x = [[np.sin(k+j)] for j in range(0,n)]
        y = [np.sin(k+n)]

        xs.append(x)
        ys.append(y)

    train_x = np.array(xs[0:1500])
    train_y = np.array(ys[0:1500])
    test_x = np.array(xs[1500:])
    test_y = np.array(ys[1500:])

    return (train_x,train_y,test_x,test_y)

length = 10
train_x,train_y,test_x,test_y = build_data(length)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

time_step_size = length
vector_size = 1
batch_size = 10
test_size = 10

x = tf.placeholder("float",[None,length,vector_size])
y = tf.placeholder("float",[None,1])

W = tf.Variable(tf.random_normal([10,1],stddev=0.01))
B = tf.Variable(tf.random_normal([1],stddev=0.01))

def seq_predict_model(X,w,b,time_step_size,vector_size):
    # input X shape:[batch_size,time_step_size,vector_size]
    # transpose X to [time_step_size,batch_size,vector_size]
    X = tf.transpose(X,[1,0,2])
    # reshape X to [time_step_size*batch_size,vector_size]
    X = tf.reshape(X,[-1,vector_size])
    # split X,array[time_step_size],shape:[batch_size,vector_size]
    X = tf.split(X,time_step_size,0)
    cell = core_rnn_cell.BasicRNNCell(num_units=10)
    initial_state = tf.zeros([batch_size,cell.state_size])
    outputs,_states = core_rnn.static_rnn(cell,X,initial_state=initial_state
    # Linear activation
    return tf.matmul(outputs[-1],w)+b,cell.state_size

