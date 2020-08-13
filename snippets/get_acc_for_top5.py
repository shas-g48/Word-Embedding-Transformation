import tensorflow as tf
import numpy as np

a = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20],[21,22,23,24],[25,26,27,28]])

b = np.asarray([[1, 3],[4, 6]])

top1_indices = tf.gather(b, 0, axis=1)
top2_indices = tf.gather(b, 1, axis=1)

top1_values = tf.gather(a, top1_indices)
top2_values = tf.gather(a, top2_indices)
c = np.array([[5,6,7,8],[25,26,27,28]])

a_t = tf.get_variable('a_t', dtype=tf.int64, initializer=tf.constant(a))
b_t = tf.get_variable('b_t', dtype=tf.int64, initializer=tf.constant(b))
c_t = tf.get_variable('c_t', dtype=tf.int64, initializer=tf.constant(c))

#a_gr = tf.gather_nd(a_t, indices=b_t, name='indexed')
e_q1 = tf.reduce_all(tf.equal(top1_values, c_t), axis=1)
e_q2 = tf.reduce_all(tf.equal(top2_values, c_t), axis=1)
e_qsumm = tf.logical_or(e_q1, e_q2)
acc = tf.reduce_sum(tf.where(e_qsumm, tf.fill(tf.shape(e_qsumm), 1), tf.fill(tf.shape(e_qsumm), 0)))

#g = a_t[:][0][1]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print(sess.run(a_gr))
    print(sess.run(top1_indices))
    print(sess.run(top2_indices))
    print(sess.run(top1_values))
    print(sess.run(top2_values))
    print(sess.run(e_q1))
    print(sess.run(e_q2))
    print(sess.run(e_qsumm))
    print(sess.run(acc))
    #print(sess.run(g))
