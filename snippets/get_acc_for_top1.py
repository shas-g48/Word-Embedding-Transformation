import tensorflow as tf
import numpy as np

a = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])

b = np.asarray([[1],[3],[4]])

c = np.array([[3,4,9,10],[13,14,15,16],[17,18,19,20]])

a_t = tf.get_variable('a_t', dtype=tf.int64, initializer=tf.constant(a))
b_t = tf.get_variable('b_t', dtype=tf.int64, initializer=tf.constant(b))
c_t = tf.get_variable('c_t', dtype=tf.int64, initializer=tf.constant(c))

a_gr = tf.gather_nd(a_t, indices=b_t, name='indexed')
e_q = tf.reduce_all(tf.equal(a_gr, c_t), axis=1)
acc = tf.reduce_sum(tf.where(e_q, tf.fill(tf.shape(e_q), 1), tf.fill(tf.shape(e_q), 0)))

g = a_t[:][0][1]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a_gr))
    print(sess.run(e_q))
    print(sess.run(acc))
    print(sess.run(g))
