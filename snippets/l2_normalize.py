import tensorflow as tf
import numpy as np

a = tf.get_variable('a',dtype=tf.float32, initializer=tf.constant([[1.2, -3.5, 4.3]]))
a_n = tf.nn.l2_normalize(a)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    print(sess.run(a_n))
