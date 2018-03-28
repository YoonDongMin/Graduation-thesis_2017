import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
tf.set_random_seed(777) # for keep the random number constant

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

xy = np.loadtxt('DT2.csv', delimiter=',', dtype=np.int32)
sale = np.loadtxt('Sale_Data.csv', delimiter=',')

x_data = xy[0:500, [0]]
y_data = xy[0:500, [-1]]
x_test = xy[500:700, [0]]
y_real = sale[500:700]
real = sale[0:500]

x_data = MinMaxScaler(x_data)
y_data = MinMaxScaler(y_data)
x_test = MinMaxScaler(x_test)

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

def test_re(data):
    print(sess.run(hypothesis * (48 + 1e-7), feed_dict={X: data}))
    return sess.run(hypothesis * (48 + 1e-7), feed_dict={X:data} )

prediction_1000 = test_re(x_test)
std_1000 = y_real - prediction_1000
std_1000 = np.std(std_1000)

prediction_1000_2 = test_re(x_test)
std_1000_2 = y_real - prediction_1000_2
std_1000_2 = np.std(std_1000_2)

prediction_10000_2 = test_re(x_test)
std_10000_2 = y_real - prediction_10000_2
std_10000_2 = np.std(std_10000_2)

prediction_50000_2 = test_re(x_test)
std_50000_2 = y_real - prediction_50000_2
std_50000_2 = np.std(std_50000_2)

plt.plot(y_real, label='Sale Data')
plt.plot(prediction_1000, label='prediction_1,000_step')
plt.plot(prediction_10000_2, label='prediction_10,000_step')
plt.plot(prediction_50000_2, label='prediction_50,000_step')
plt.legend()
plt.show()