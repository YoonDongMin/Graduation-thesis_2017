import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
tf.set_random_seed(777)  # for keep the random number constant

xy = np.loadtxt('DT1.csv', delimiter=',', dtype=np.int32) # Order Data load
sale = np.loadtxt('Sale_Data.csv', delimiter=',') # Sale Data load

x_data = xy[0:500, [0]]
y_data = xy[0:500, [-1]]
x_test = xy[500:700, [0]]
y_real = sale[500:700]

X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([1, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(1000):
    cost_val, hy_val, acc, _ = sess.run(
        [cost, hypothesis, accuracy, train], feed_dict={X: x_data, Y: y_data})
    print(step, "Cost: ", cost_val,  "\nAccuracy:\n", acc)

def test_hy(data):
    return sess.run(hypothesis, feed_dict={X:data})

BT = y_data - y_real
BT =  np.std(BT) # Standard Deviation

prediction_1000 = test_hy(x_test)
std_1000 = y_real - prediction_1000
std_1000 = np.std(std_1000)

prediction_10000 = test_hy(x_test)
std_10000 = y_real - prediction_10000
std_10000 = np.std(std_10000)

prediction_50000 = test_hy(x_test)
std_50000 = y_real - prediction_50000
std_50000 = np.std(std_50000)

prediction_100000 = test_hy(x_test)
std_100000 = y_real - prediction_100000
std_100000 = np.std(std_100000)

prediction_200000 = test_hy(x_test)
std_200000 = y_real - prediction_200000
std_200000 = np.std(std_200000)

prediction_500000 = test_hy(x_test)
std_500000 = y_real - prediction_500000
std_500000 = np.std(std_500000)

plt.plot(y_real, label='Sale Data')
plt.plot(prediction_100000, label='prediction_100,000_step')
plt.plot(prediction_200000, label='prediction_200,000_step')
plt.plot(prediction_500000, label='prediction_500,000_step')
plt.legend()
plt.show()
