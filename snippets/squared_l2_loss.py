import tensorflow as tf
import numpy as np

a = [[2.,3.,4.],[5.,6.,7.]]
b = [[9.,10.,11.],[13.,14.,17.]]

a_n = np.asarray(a, dtype=np.float32)
b_n = np.asarray(b, dtype=np.float32)

a_t = tf.get_variable('a_t',dtype=tf.float32,initializer=tf.constant(a_n))
b_t = tf.get_variable('b_t', dtype=tf.float32,initializer=tf.constant(b_n))

g = tf.math.square(a_t - b_t)
#g1 = tf.reduce_sum(g, 0)
g1 = tf.reduce_sum(g, 1)
g2 = tf.reduce_mean(g1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(g))
    print(sess.run(g1))
    print(sess.run(g2))

