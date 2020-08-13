import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import tensorflow as tf
from math import ceil

import utils

class TranslateLinearTransform(object):

    def __init__(self):
        self.lr = 1e-3
        self.gstep = tf.get_variable('gstep', dtype=tf.int32, initializer=tf.constant(0))
        self.en = utils.load_vectors('en.vec')
        self.es = utils.load_vectors('es.vec')
        self.batch_size = 16    # batch size for training
        self.batch_size_test = 100
        self.batch_size_partb = 9
        self.skip_step = 500    # to print loss at regular intervals
        self.size = 2124    # training dataset size
        self.size_test = 235
        self.en_vec = tf.get_variable('en_vec', dtype=tf.float32,initializer=tf.constant(list(self.en.values())))
        self.top1_total = 0
        self.top5_total = 0

    def get_data(self):
        """ Load the dataset, and make initializable iterators """
        with tf.name_scope('data'):
            train_data = utils.get_data('word_transform/train.vocab', self.batch_size, self.en, self.es)
            test_data = utils.get_data('word_transform/eval.vocab', self.batch_size_test, self.en, self.es)
            partb_data = utils.get_data('word_transform/partb.vocab', self.batch_size_partb, self.en, self.es)

            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                    train_data.output_shapes)
            self.input, self.output = iterator.get_next()

            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)
            self.partb_init = iterator.make_initializer(partb_data)

    def inference(self):
        """ Define the inference part of graph """
        with tf.name_scope('inference'):
            W = tf.get_variable('transformation_matrix', dtype=tf.float32, shape=(300,300), initializer=tf.truncated_normal_initializer())
            self.trans = tf.matmul(W, tf.transpose(self.input))

    def loss(self):
        """
        Defines loss used to train
        loss is squared l2 norm of difference between output and trans
        reduce_mean to take mean over the batch size
        """
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(self.trans - tf.transpose(self.output)), 1))


    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                    global_step=self.gstep)

    def summary(self):
        """ Define summaries for loss, top1 and top5 accuracy """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('top1_accuracy', self.top1_total)
            tf.summary.scalar('top5_accuracy', self.top5_total)
            self.summary_op = tf.summary.merge_all()

    def evaluate(self):
        """ Define operations required to evaluate top1 and top5 accuracy """
        with tf.name_scope('evaluate'):
            self.top1 = tf.get_variable('top1', initializer=tf.constant(0))
            self.top5 = tf.get_variable('top5', initializer=tf.constant(0))

            # Placeholders to feed data in while evaluating
            self.raw_trans = tf.placeholder(dtype=tf.float32, shape=(300,None))
            self.raw_output = tf.placeholder(dtype=tf.float32, shape=(300, None))

            # normalize both en vocab and raw trans to compute cosine similarity
            # axis = 0 for trans_normalized as self.raw_trans is of dims (embed, batch_size_test or ?)
            trans_normalized = tf.nn.l2_normalize(self.raw_trans, axis=0)
            # axis = 1 for vector_normalized as self.en_vec is of dims (vocab, embed)
            vector_normalized = tf.nn.l2_normalize(self.en_vec, axis=1)
            # cosine similarity is of dims (vocab, batch_size_test or ?)
            cosine_similarity = tf.matmul(vector_normalized, trans_normalized)

            # dims (100, 1) ie (batch_size_test, k)
            # need transpose as tf.nn.top_k finds top_k in a row
            self.top1_values, self.top1_indices = tf.nn.top_k(tf.transpose(cosine_similarity), k=1)
            # dims (100, 5) ie (batch_size_test, k)
            self.top5_values, self.top5_indices = tf.nn.top_k(tf.transpose(cosine_similarity), k=5)

            output_trans = tf.transpose(self.raw_output)

            # get vectors corresponding to top1 cosine accuracy
            top1_gr = tf.gather_nd(self.en_vec, indices=self.top1_indices, name='top1_indexed')
            # check if equals correct output
            # reduce to get one boolean for each word
            top1_eq = tf.reduce_all(tf.equal(top1_gr, output_trans), axis=1)
            # convert from booleans to scalar, by replacing true with 1 and
            # false by zero then summing them
            top1_batch = tf.reduce_sum(tf.where(top1_eq, tf.fill(tf.shape(top1_eq), 1), tf.fill(tf.shape(top1_eq), 0)))
            self.top1_upd = self.top1.assign_add(top1_batch)

            # get 5 index vectors of dims (100,1) from (100, 5) by separating cols
            top5_indices1 = tf.gather(self.top5_indices, 0, axis=1)
            top5_indices2 = tf.gather(self.top5_indices, 1, axis=1)
            top5_indices3 = tf.gather(self.top5_indices, 2, axis=1)
            top5_indices4 = tf.gather(self.top5_indices, 3, axis=1)
            top5_indices5 = tf.gather(self.top5_indices, 4, axis=1)

            # gather vectors corresponding to each for comparison with output
            top5_values1 = tf.gather(self.en_vec, top5_indices1)
            top5_values2 = tf.gather(self.en_vec, top5_indices2)
            top5_values3 = tf.gather(self.en_vec, top5_indices3)
            top5_values4 = tf.gather(self.en_vec, top5_indices4)
            top5_values5 = tf.gather(self.en_vec, top5_indices5)

            # compare with output all 5 embed matrices and then reduce to boolean
            # for each word
            top5_eq1 = tf.reduce_all(tf.equal(top5_values1, output_trans), axis=1)
            top5_eq2 = tf.reduce_all(tf.equal(top5_values2, output_trans), axis=1)
            top5_eq3 = tf.reduce_all(tf.equal(top5_values3, output_trans), axis=1)
            top5_eq4 = tf.reduce_all(tf.equal(top5_values4, output_trans), axis=1)
            top5_eq5 = tf.reduce_all(tf.equal(top5_values5, output_trans), axis=1)

            # do or over all the top 5 matrices
            top5_eq_1_2 = tf.logical_or(top5_eq1, top5_eq2)
            top5_eq_3_4 = tf.logical_or(top5_eq3, top5_eq4)
            top5_eq_12_34 = tf.logical_or(top5_eq_1_2, top5_eq_3_4)
            top5_eq = tf.logical_or(top5_eq_12_34, top5_eq5)

            # compute scalar acc from the final top5_eq, 'or' of all
            top5_batch = tf.reduce_sum(tf.where(top5_eq, tf.fill(tf.shape(top5_eq), 1), tf.fill(tf.shape(top5_eq), 0)))
            self.top5_upd = self.top5.assign_add(top5_batch)

            # used to reset top1 & top5 before each epoch evaluation
            self.top1_reset = tf.assign(self.top1, tf.constant(0))
            self.top5_reset = tf.assign(self.top5, tf.constant(0))

    def eval_once(self, sess):
        """ Runs the evaluation and updates top1_total and top5_total """
        # reset top1_total and top5_total and initialize iterator for test
        sess.run([self.top1_reset, self.top5_reset])
        sess.run(self.test_init)

        top1 = 0
        top5 = 0
        try:
            while True:
                # get output and computes translation to feed in to evaluate
                raw_trans, raw_output = sess.run([self.trans, self.output])
                top1, top5 = sess.run([self.top1_upd, self.top5_upd], feed_dict={self.raw_trans:raw_trans, self.raw_output:np.transpose(raw_output)})

        except tf.errors.OutOfRangeError:
            pass

        # convert to percentage
        self.top1_total = (top1 / self.size_test) * 100
        self.top5_total = (top5 / self.size_test) * 100

    def partb_inference(self):
        """ Defines operations needed to get top1 and top5 translations for each word """
        # Placeholder to feed in the raw translated vector for each input
        self.partb_raw_trans = tf.placeholder(dtype=tf.float32, shape=(300, None))

        # find cosine similarity between vocab and raw translation
        trans_normalized = tf.nn.l2_normalize(self.partb_raw_trans, axis=0)
        vector_normalized = tf.nn.l2_normalize(self.en_vec, axis=1)
        cosine_similarity = tf.matmul(vector_normalized, trans_normalized)

        # get the indices for top1 and top5 translations respectively
        _, top1_indices = tf.nn.top_k(tf.transpose(cosine_similarity), k=1)
        _, top5_indices = tf.nn.top_k(tf.transpose(cosine_similarity), k=5)

        # get vectors for top1
        top1_gr = tf.gather_nd(self.en_vec, indices=top1_indices, name='top1_inidices')

        # get 5 index vectors of dims (100,1) from (100, 5) by separating cols
        top5_indices1 = tf.gather(top5_indices, 0, axis=1)
        top5_indices2 = tf.gather(top5_indices, 1, axis=1)
        top5_indices3 = tf.gather(top5_indices, 2, axis=1)
        top5_indices4 = tf.gather(top5_indices, 3, axis=1)
        top5_indices5 = tf.gather(top5_indices, 4, axis=1)

        # gather vectors corresponding to each for comparison with output
        top5_values1 = tf.gather(self.en_vec, top5_indices1)
        top5_values2 = tf.gather(self.en_vec, top5_indices2)
        top5_values3 = tf.gather(self.en_vec, top5_indices3)
        top5_values4 = tf.gather(self.en_vec, top5_indices4)
        top5_values5 = tf.gather(self.en_vec, top5_indices5)

        # values to be retrieved by driver function
        self.partb_top1 = top1_gr
        self.partb_top51 = top5_values1
        self.partb_top52 = top5_values2
        self.partb_top53 = top5_values3
        self.partb_top54 = top5_values4
        self.partb_top55 =  top5_values5

    def partb_eval(self, sess):
        """
        Get top1 and top5 vectors from partb_inference
        convert to words and write to file (translation.txt)
        """

        sess.run(self.partb_init)
        raw_trans = sess.run([self.trans])

        # get the top1 and top5 vectors
        partb_top1, partb_top51, partb_top52, partb_top53, partb_top54, partb_top55 = sess.run([self.partb_top1, self.partb_top51, self.partb_top52, self.partb_top53, self.partb_top54, self.partb_top55], feed_dict={self.partb_raw_trans:raw_trans[0]})

        # lists to hold the words
        top1_words = []
        top51_words =[]
        top52_words =[]
        top53_words =[]
        top54_words =[]
        top55_words =[]

        # for each word vector get it's corresponding word from en vec
        start_time = time.time()
        for i in partb_top1:
            for key, val in self.en.items():
                # data in i is 32 bit, so need 32 bit vals for comparisons
                # note: converting i to 64 bit won't work, as cannot increase
                # precision of a lower precision number
                if (np.asarray(i)==np.asarray(val, dtype=np.float32)).all():
                    top1_words.append(key)

        print(top1_words)
        print("#debug took time: ", time.time() - start_time)

        start_time = time.time()
        for i in partb_top51:
            for key, val in self.en.items():
                if (np.asarray(i)==np.asarray(val, dtype=np.float32)).all():
                    top51_words.append(key)

        print(top51_words)
        print("#debug took time: ", time.time() - start_time)

        start_time = time.time()
        for i in partb_top52:
            for key, val in self.en.items():
                if (np.asarray(i)==np.asarray(val, dtype=np.float32)).all():
                    top52_words.append(key)

        print(top52_words)
        print("#debug took time: ", time.time() - start_time)

        start_time = time.time()
        for i in partb_top53:
            for key, val in self.en.items():
                if (np.asarray(i)==np.asarray(val, dtype=np.float32)).all():
                    top53_words.append(key)

        print(top53_words)
        print("#debug took time: ", time.time() - start_time)

        start_time = time.time()
        for i in partb_top54:
            for key, val in self.en.items():
                if (np.asarray(i)==np.asarray(val, dtype=np.float32)).all():
                    top54_words.append(key)

        print(top54_words)
        print("#debug took time: ", time.time() - start_time)

        start_time = time.time()
        for i in partb_top55:
            for key, val in self.en.items():
                if (np.asarray(i)==np.asarray(val, dtype=np.float32)).all():
                    top55_words.append(key)

        print(top55_words)
        print("#debug took time: ", time.time() - start_time)

        # write the results to file
        utils.write_translation('translation.txt', self.batch_size_partb, top1_words, top51_words, top52_words, top53_words, top54_words, top55_words)

    def build(self):
        """ Build the model """
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.evaluate()
        self.summary()

        self.partb_inference()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        total_loss = 0
        n_batches = ceil(self.size/self.batch_size)

        try:
            while True:
                t, _, l, summaries = sess.run([self.trans, self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)

                # print loss at regular intervals
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))

                step += 1
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, 'checkpoints/translate/translate', step)
        # Print average loss for epoch
        print('Average loss at epoch {0}:{1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))

        return step

    def train(self, n_epochs):
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/translate')
        writer = tf.summary.FileWriter('./graphs/translate', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/translate/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)

                # evaluate after every epoch of training
                self.eval_once(sess)
                print('Accuracy top1 at epoch {}: {}'.format(epoch, self.top1_total))
                print('Accuracy top5 at epoch {}: {}'.format(epoch, self.top5_total))

            # to get the final top1 and top5 acc, mean over 10 evaluations is taken
            final_top1 = 0
            final_top5 = 0
            for i in range(10):
                self.eval_once(sess)
                final_top1 += self.top1_total
                final_top5 += self.top5_total

            print("Final Accuracy top1 (Mean of 10 evaluations): ", final_top1/10)
            print("Final Accuracy top5 (Mean of 10 evaluations): ", final_top5/10)

            self.partb_eval(sess)

        writer.close()

if __name__ == '__main__':
    model = TranslateLinearTransform()
    model.build()
    print('#debug: model built!')
    model.train(n_epochs=170)
