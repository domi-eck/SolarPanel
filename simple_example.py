import tensorflow as tf
slim = tf.contrib.slim
import numpy as np


node1 = tf.placeholder(dtype=tf.float32, shape=())
node2 = tf.placeholder(dtype=tf.float32, shape=())  # also tf.float32 implicitly
# And define some operation on these nodes
node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
# Only now we instantiate the graph and execute it

init_op = tf.global_variables_initializer()

my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()


with tf.Session() as sess:
    res = sess.run(node4, feed_dict={node1: 10, node2: 2})
    print(res)

    while True:
        try:
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            break




