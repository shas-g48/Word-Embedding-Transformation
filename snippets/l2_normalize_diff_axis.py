import tensorflow as tf
import numpy as np

b = np.asarray([[1,2],[3,4],[5,6]], dtype=np.float32)
a = tf.get_variable('a', dtype=tf.float32, initializer=tf.constant(b))
n1 = tf.nn.l2_normalize(a, axis=0)
n2 = tf.nn.l2_normalize(a, axis=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(n1))
    print(sess.run(n2))
