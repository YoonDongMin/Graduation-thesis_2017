import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for keep the random number constant
import pprint
from openpyxl import load_workbook
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

xy = np.loadtxt('DT1.csv', delimiter=',', dtype=np.int32)
sale = np.loadtxt('Sale_Data.csv', delimiter=',')

X_data = xy[0:500, [0]]
N = X_data.shape[0]
y_data = xy[0:500, [-1]]
nb_classes = max(np.hstack(y_data).tolist()) + 1
x_test = xy[500:700, [0]]
y_real = sale[500:700]

learning_rate = 0.01
training_epochs = 30
batch_size = 100

X = tf.placeholder(tf.float32, [None, 501])
Y = tf.placeholder(tf.float32, [None, 49])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[501, 100],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([100]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[100, 49],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([49]))
hypothesis = tf.matmul(L1, W2) + b2

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    avg_acc = 0
    total_batch = int( batch_size)

    for i in range(total_batch):
        nb = max(np.hstack(y_data).tolist()) + 1
        targets_X = X_data.reshape(-1)
        targets_Y = y_data.reshape(-1)
        one_hot_X = np.eye(len(y_data)+1)[targets_X].tolist()
        one_hot_X = np.reshape(one_hot_X, (500, 501))
        one_hot_y = np.eye(nb)[targets_Y]
        feed_dict = {X: one_hot_X, Y: one_hot_y, keep_prob: 1}
        c, a, _ = sess.run([cost, accuracy, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        avg_acc += a / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('acc = ', '{:.7f}'.format(avg_acc))
print('Learning Finished!')

def prediction(data, size):
    return sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: data, keep_prob: size})

def prediction(data, size):
    targets_X = data.reshape(-1)  # len is 3500
    one_hot_X = np.eye(len(y_data) + 1)[targets_X].tolist()
    one_hot_X = np.reshape(one_hot_X, (len(data), len(X_data) * X_data.shape[-1] + X_data.shape[-1]))
    return sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: one_hot_X, keep_prob: size})

BT = y_data - y_real
BT =  np.std(BT) # Standard Deviation

prediction_10 = prediction(x_test, 1)
std_10 = y_real - prediction_10
std_10 = np.std(std_10)

prediction_20 = prediction(x_test, 1)
std_20 = y_real - prediction_20
std_20 = np.std(std_20)

plt.plot(y_real, label='Sale Data')
plt.plot(prediction_10, label='prediction_100,000_step')
plt.legend()
plt.show()