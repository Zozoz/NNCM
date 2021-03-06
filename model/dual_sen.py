#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from newbie_nn.config import *
from newbie_nn.nn_layer import cnn_layer, bi_dynamic_rnn, softmax_layer, reduce_mean_with_len
from newbie_nn.att_layer import mlp_attention_layer
from data_prepare.utils import load_w2v, batch_index, load_inputs_twitter, load_inputs_document_nohn

tf.app.flags.DEFINE_float('alpha', 0.6, 'learning rate')
tf.app.flags.DEFINE_string('embedding_file_path_o', '', 'embedding file path')
tf.app.flags.DEFINE_string('embedding_file_path_r', '', 'embedding file path')


def bi_rnn(inputs, sen_len, keep_prob1, keep_prob2, _id='1'):
    print 'I am bi-rnn.'
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    cell = tf.nn.rnn_cell.LSTMCell
    hiddens = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sentence' + _id, FLAGS.t1)
    return softmax_layer(hiddens, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class, _id)


def pooling(inputs, dim=1, _type='max'):
    if _type == 'max':
        return tf.reduce_max(inputs, dim, False)
    elif _type == 'avg':
        return tf.reduce_mean(inputs, dim, False)


def cnn(inputs, sen_len, keep_prob1, keep_prob2, _id='1'):
    print 'I am cnn.'
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    inputs = tf.reshape(inputs, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim, 1])

    conv1 = cnn_layer(inputs, [3, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base,
                      FLAGS.l2_reg, scope_name='conv1'+_id)
    conv1 = tf.reshape(conv1, [-1, FLAGS.max_sentence_len - 2, FLAGS.n_hidden])
    conv1 = pooling(conv1, 1, 'max')

    conv2 = cnn_layer(inputs, [2, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base,
                      FLAGS.l2_reg, scope_name='conv2'+_id)
    conv2 = tf.reshape(conv2, [-1, FLAGS.max_sentence_len - 1, FLAGS.n_hidden])
    conv2 = pooling(conv2, 1, 'max')

    conv3 = cnn_layer(inputs, [1, FLAGS.embedding_dim, 1, FLAGS.n_hidden], [1, 1, 1, 1], 'VALID', FLAGS.random_base,
                      FLAGS.l2_reg, scope_name='conv3'+_id)
    conv3 = tf.reshape(conv3, [-1, FLAGS.max_sentence_len - 0, FLAGS.n_hidden])
    conv3 = pooling(conv3, 1, 'max')

    outputs = (conv1 + conv2 + conv3) / 3.

    return softmax_layer(outputs, FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class, _id)


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path_o, FLAGS.embedding_dim, True)
    word_embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')

    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x')
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='y')
        sen_len = tf.placeholder(tf.int32, [None], name='sen_len')

    inputs = tf.nn.embedding_lookup(word_embedding, x)

    if FLAGS.method == 'CNN':
        prob = cnn(inputs, sen_len, keep_prob1, keep_prob2)
    else:
        prob = bi_rnn(inputs, sen_len, keep_prob1, keep_prob2)

    loss = loss_func(y, prob)
    acc_num, acc_prob = acc_func(y, prob)
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = train_func(loss, FLAGS.learning_rate, global_step)

    title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
        FLAGS.keep_prob1,
        FLAGS.keep_prob2,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.l2_reg,
        FLAGS.max_sentence_len,
        FLAGS.embedding_dim,
        FLAGS.n_hidden,
        FLAGS.n_class
    )

    with tf.Session() as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        saver = saver_func(save_dir)

        init = tf.initialize_all_variables()
        sess.run(init)

        tr_x, tr_sen_len, tr_y = load_inputs_twitter(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len
        )
        te_x, te_sen_len, te_y = load_inputs_twitter(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len
        )

        def get_batch_data(xi, sen_leni, yi, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: xi[index],
                    y: yi[index],
                    sen_len: sen_leni[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc, max_prob, step = 0, None, None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_y, FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)

            acc, cost, cnt = 0., 0., 0
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_y, 2000, 1.0, 1.0, False):
                _loss, _acc, _p = sess.run([loss, acc_num, prob], feed_dict=test)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
            print 'all samples={}, correct prediction={}'.format(cnt, acc)
            acc = acc / cnt
            cost = cost / cnt
            print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, cost, acc)
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            if acc > max_acc:
                max_acc = acc
                max_prob = p

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')

        print 'Optimization Finished! Max acc={}'.format(max_acc)
        print 'Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        )


def main_(_):
    word_id_mapping_o, w2v_o = load_w2v(FLAGS.embedding_file_path_o, FLAGS.embedding_dim, True)
    word_embedding_o = tf.constant(w2v_o, dtype=tf.float32)
    word_id_mapping_r, w2v_r = load_w2v(FLAGS.embedding_file_path_r, FLAGS.embedding_dim, True)
    word_embedding_r = tf.constant(w2v_r, dtype=tf.float32)

    with tf.name_scope('inputs'):
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)
        x_o = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        x_r = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        len_o = tf.placeholder(tf.int32, [None])
        len_r = tf.placeholder(tf.int32, [None])
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])

    with tf.device('/gpu:0'):
        inputs_o = tf.nn.embedding_lookup(word_embedding_o, x_o)
        inputs_o = tf.reshape(inputs_o, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])
        if FLAGS.method == 'CNN':
            prob_o = cnn(inputs_o, len_o, keep_prob1, keep_prob2, 'o')
        else:
            prob_o = bi_rnn(inputs_o, len_o, keep_prob1, keep_prob2, 'o')
    with tf.device('/gpu:1'):
        inputs_r = tf.nn.embedding_lookup(word_embedding_r, x_r)
        inputs_r = tf.reshape(inputs_r, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])
        if FLAGS.method == 'CNN':
            prob_r = cnn(inputs_r, len_r, keep_prob1, keep_prob2, 'r')
        else:
            prob_r = bi_rnn(inputs_r, len_r, keep_prob1, keep_prob2, 'r')

    r_y = tf.reverse(y, [False, True])
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y * tf.log(prob_o)) - tf.reduce_mean(r_y * tf.log(prob_r)) + sum(reg_loss)
    prob = FLAGS.alpha * prob_o + (1.0 - FLAGS.alpha) * tf.reverse(prob_r, [False, True])

    acc_num, acc_prob = acc_func(y, prob)
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = train_func(loss, FLAGS.learning_rate, global_step)

    title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
        FLAGS.keep_prob1,
        FLAGS.keep_prob2,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.l2_reg,
        FLAGS.max_sentence_len,
        FLAGS.embedding_dim,
        FLAGS.n_hidden,
        FLAGS.n_class
    )

    conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=conf) as sess:
        import time
        timestamp = str(int(time.time()))
        _dir = 'summary/' + str(timestamp) + '_' + title
        test_loss = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
        validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

        save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
        saver = saver_func(save_dir)

        init = tf.initialize_all_variables()
        sess.run(init)

        # saver.restore(sess, '/-')

        tr_x, tr_y, tr_sen_len = load_inputs_document_nohn(
            FLAGS.train_file_path,
            word_id_mapping_o,
            FLAGS.max_sentence_len
        )
        te_x, te_y, te_sen_len = load_inputs_document_nohn(
            FLAGS.test_file_path,
            word_id_mapping_o,
            FLAGS.max_sentence_len
        )
        tr_x_r, tr_y_r, tr_sen_len_r = load_inputs_document_nohn(
            FLAGS.train_file_path_r,
            word_id_mapping_r,
            FLAGS.max_sentence_len
        )
        te_x_r, te_y_r, te_sen_len_r = load_inputs_document_nohn(
            FLAGS.test_file_path_r,
            word_id_mapping_r,
            FLAGS.max_sentence_len
        )

        def get_batch_data(xo, slo, xr, slr, yy, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yy), batch_size, 1, is_shuffle):
                feed_dict = {
                    x_o: xo[index],
                    x_r: xr[index],
                    y: yy[index],
                    len_o: slo[index],
                    len_r: slr[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_prob = None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_x_r, tr_sen_len_r, tr_y,
                                           FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)

            acc, cost, cnt = 0., 0., 0
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_r, te_sen_len_r, te_y, 2000, 1.0, 1.0, False):
                _loss, _acc, _p = sess.run([loss, acc_num, prob], feed_dict=test)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
            print 'all samples={}, correct prediction={}'.format(cnt, acc)
            acc = acc / cnt
            cost = cost / cnt
            print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, cost, acc)
            summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
            test_summary_writer.add_summary(summary, step)
            if acc / cnt > max_acc:
                max_acc = acc / cnt
                max_prob = p

        fp = open(FLAGS.prob_file, 'w')
        for item in max_prob:
            fp.write(' '.join([str(it) for it in item]) + '\n')
        print 'Optimization Finished! Max acc={}'.format(max_acc)

        print 'Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
            FLAGS.learning_rate,
            FLAGS.n_iter,
            FLAGS.batch_size,
            FLAGS.n_hidden,
            FLAGS.l2_reg
        )


if __name__ == '__main__':
    tf.app.run()
