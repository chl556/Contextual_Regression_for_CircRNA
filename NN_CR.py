from Functions import *

import tensorflow as tf
import tensorflow.python.ops.rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.metrics_impl import auc
from tensorflow.python.ops.nn_impl import batch_normalization
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.nn_ops import relu6
from tensorflow.python.ops.nn_ops import relu
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import stack
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.layers.normalization import batch_normalization
from tensorflow.contrib.metrics import confusion_matrix

filename = sys.argv[1]
reg_scale = float(sys.argv[2])
load_list = bool(int(sys.argv[3]))

# read data
motif_data = read_tsv(filename)
di, processed_x, processed_y = get_map(motif_data)

# scale x to Gaussian
scaler = StandardScaler()
processed_x = scaler.fit_transform(processed_x)

np.save("mean", scaler.mean_)
np.save("scale", scaler.scale_)

# setup the variables in neural network
hidden_size_1 = 10
hidden_size_2 = 10
hidden_size_3 = 10
num_feat = len(processed_x[0])
num_type = len(di)
batch_size = 50

# setup the hyper-parameters
max_grad_norm = 10
learning_rate = 0.0001

# ratio of data division
train_percent = 0.7

# number of training epochs
num_of_epoch = 200

# neural network section
# ------------------------------------------------------------------------------

x = tf.placeholder(tf.float32, shape=[None, num_feat])
y_ = tf.placeholder(tf.float32, shape=[None, num_type])
is_training = tf.placeholder(tf.bool)

# batch normalization the x
x_bn = batch_norm(x, is_training=is_training)

# initial weights generator functions


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.05, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape, name=name)
    return tf.Variable(initial)


# parameter declaration
w_1 = weight_variable(name="w_1", shape=[num_feat, hidden_size_1])
b_1 = bias_variable(name="b_1", shape=[hidden_size_1])
w_2 = weight_variable(name="w_2", shape=[hidden_size_1, hidden_size_2])
b_2 = bias_variable(name="b_2", shape=[hidden_size_2])
w_3 = weight_variable(name="w_3", shape=[hidden_size_2, hidden_size_3])
b_3 = bias_variable(name="b_3", shape=[hidden_size_3])
w_4 = weight_variable(name="w_4", shape=[hidden_size_3, num_feat])
b_4 = bias_variable(name="b_4", shape=[num_feat])
b = bias_variable(name="b", shape=[1])

# stacking of 3 layers of resnet
layer_1 = tanh(tf.add(tf.matmul(x_bn, w_1), b_1))
layer_2 = tanh(tf.add(tf.matmul(layer_1, w_2), b_2)) + layer_1
layer_3 = tanh(tf.add(tf.matmul(layer_2, w_3), b_3)) + layer_2

# calculate context and feature contribution
context = tf.add(tf.matmul(layer_3, w_4), b_4)
contribution = tf.multiply(context, x_bn)

# dot product that apply context
prob_pre_logregress = tf.reduce_sum(contribution, axis=1) + b
# 0 is circular and 1 is noncircular, prob is the prob for 0
prob = tf.expand_dims(input=sigmoid(prob_pre_logregress), axis=1)
prob_1_ = tf.constant(1.0) - prob
# concatenate to form y in 1 hot encoding
y = tf.concat(values=[prob, prob_1_], axis=1)

# calculate the class encoding of y
int_y_ = tf.argmax(y_, 1)
int_y = tf.argmax(y, 1)

# concatenate into an array for analysis: [features, contributions, prob of class, pre_label, real_label]
output_for_analysis = tf.concat(values=[x, contribution, prob_1_, tf.cast(tf.expand_dims(
    int_y, axis=1), dtype=tf.float32), tf.cast(tf.expand_dims(int_y_, axis=1), dtype=tf.float32)], axis=1)

# calculate cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(y), reduction_indices=[1]))

# apply lasso penalty for weight sparsity
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=reg_scale)

# calculate costs
error_cost = cross_entropy
regularization_cost = tf.contrib.layers.apply_regularization(l1_regularizer, [
                                                             context])
cost = error_cost + regularization_cost

# training
tvars = tf.trainable_variables()
gradi = tf.gradients(cost, tvars)
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.apply_gradients(zip(grads, tvars))

# ------------------------------------------------------------------------------

train_num = int(train_percent * len(processed_y))

print("Whether to use load_list:")
print(load_list)

# divide the training and testing set
if load_list:
    train_list = np.loadtxt("train_list", dtype=np.int32)
    test_list = np.loadtxt("test_list", dtype=np.int32)
else:
    # separate data
    index_list = list(range(0, len(processed_y)))
    random.shuffle(index_list)

    train_list = index_list[:train_num]
    test_list = index_list[train_num:]

    np.savetxt("train_list", train_list)
    np.savetxt("test_list", test_list)

train_x = obtain_data_from_list(processed_x, train_list)
train_y = obtain_data_from_list(processed_y, train_list)
test_x = obtain_data_from_list(processed_x, test_list)
test_y = obtain_data_from_list(processed_y, test_list)

# calculate steps per epoch
steps_per_epoch = int(train_num / batch_size)

# create saver
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    for epoch in range(num_of_epoch + 1):
        start = time.time()
        print("Epoch %d: " % epoch)

        # train
        for i in range(steps_per_epoch):
            # train
            x_t, y_t = get_batch(train_x, train_y, batch_size)
            train_step.run(feed_dict={x: x_t, y_: y_t, is_training: True})

        end = time.time()
        print("this epoch takes a total of: " +
              str(end - start) + " seconds \n")
            
        # calculate prediction with the current model
        predicted_y_train = sess.run(
            y, feed_dict={x: train_x, y_: train_y, is_training: False})
        predicted_y_test = sess.run(
            y, feed_dict={x: test_x, y_: test_y, is_training: False})

        # output auc
        print("roc auc train:", roc_auc_score(train_y, predicted_y_train))
        print("roc auc test:", roc_auc_score(test_y, predicted_y_test))

        train_one_hot = np.round(predicted_y_train)
        test_one_hot = np.round(predicted_y_test)

        train_class = np.argmax(train_one_hot, axis=1)
        test_class = np.argmax(test_one_hot, axis=1)

        train_true = np.argmax(train_y, axis=1)
        test_true = np.argmax(test_y, axis=1)

        # output accuracy
        print("train accuracy:", accuracy_score(train_true, train_class))
        print("test accuracy:", accuracy_score(test_true, test_class))

        # save every 20 epochs
        if epoch % 20 == 0:
            train_result = sess.run(output_for_analysis, feed_dict={
                                    x: train_x, y_: train_y, is_training: False})
            test_result = sess.run(output_for_analysis, feed_dict={
                                   x: test_x, y_: test_y, is_training: False})

            combined_result = np.concatenate(
                [train_result, test_result], axis=0)
            print(combined_result.shape)

            np.save("result_at_epoch_" + str(epoch), combined_result)

            save_path = saver.save(sess, "./model_" + str(epoch) + ".ckpt")
            print("save model at " + str(epoch))
