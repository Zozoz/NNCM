#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from newbie_nn.config import *
from newbie_nn.nn_layer import dynamic_rnn, bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import mlp_attention_layer
from data_prepare.utils import load_w2v, batch_index, load_word_embedding, load_inputs_document

tf.app.flags.DEFINE_float('alpha', 0.6, 'learning rate')
tf.app.flags.DEFINE_string('embedding_file_path_o', '', 'embedding file path')
tf.app.flags.DEFINE_string('embedding_file_path_r', '', 'embedding file path')


def hn_att(inputs, sen_len, doc_len, keep_prob1, keep_prob2, _id='1'):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    cell = tf.nn.rnn_cell.LSTMCell
    sen_len = tf.reshape(sen_len, [-1])
    hiddens_sen = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sentence' + _id, 'all')
    alpha_sen = mlp_attention_layer(hiddens_sen, sen_len, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'sentence' + _id)
    outputs_sen = tf.reshape(tf.batch_matmul(alpha_sen, hiddens_sen), [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])

    hiddens_doc = bi_dynamic_rnn(cell, outputs_sen, FLAGS.n_hidden, doc_len, FLAGS.max_doc_len, 'doc' + _id, 'all')
    alpha_doc = mlp_attention_layer(hiddens_doc, doc_len, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'doc' + _id)
    outputs_doc = tf.reshape(tf.batch_matmul(alpha_doc, hiddens_doc), [-1, 2 * FLAGS.n_hidden])
    return outputs_doc


def hn(inputs, sen_len, doc_len, keep_prob1, keep_prob2, _id='1'):
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    cell = tf.nn.rnn_cell.LSTMCell
    sen_len = tf.reshape(sen_len, [-1])
    hiddens_sen = bi_dynamic_rnn(cell, inputs, FLAGS.n_hidden, sen_len, FLAGS.max_sentence_len, 'sentence' + _id, FLAGS.t1)
    hiddens_sen = tf.reshape(hiddens_sen, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
    hidden_doc = bi_dynamic_rnn(cell, hiddens_sen, FLAGS.n_hidden, doc_len, FLAGS.max_doc_len, 'doc' + _id, FLAGS.t2)
    return hidden_doc


def main(_):
    word_id_mapping_o, w2v_o = load_w2v(FLAGS.embedding_file_path_o, FLAGS.embedding_dim, True)
    word_embedding_o = tf.constant(w2v_o, dtype=tf.float32)
    word_id_mapping_r, w2v_r = load_w2v(FLAGS.embedding_file_path_r, FLAGS.embedding_dim, True)
    word_embedding_r = tf.constant(w2v_r, dtype=tf.float32)

    with tf.name_scope('inputs'):
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)
        x_o = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sentence_len])
        x_r = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sentence_len])
        sen_len_o = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
        sen_len_r = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
        doc_len_o = tf.placeholder(tf.int32, None)
        doc_len_r = tf.placeholder(tf.int32, None)
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])

    with tf.device('/gpu:0'):
        inputs_o = tf.nn.embedding_lookup(word_embedding_o, x_o)
        inputs_o = tf.reshape(inputs_o, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])
        if FLAGS.method == 'ATT':
            h_o = hn_att(inputs_o, sen_len_o, doc_len_o, keep_prob1, keep_prob2, 'o')
        else:
            h_o = hn(inputs_o, sen_len_o, doc_len_o, keep_prob1, keep_prob2, 'o')
        prob_o = softmax_layer(h_o, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class, 'o')
    with tf.device('/gpu:1'):
        inputs_r = tf.nn.embedding_lookup(word_embedding_r, x_r)
        inputs_r = tf.reshape(inputs_r, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])
        if FLAGS.method == 'ATT':
            h_r = hn_att(inputs_r, sen_len_r, doc_len_r, keep_prob1, keep_prob2, 'r')
        else:
            h_r = hn(inputs_r, sen_len_r, doc_len_r, keep_prob1, keep_prob2, 'r')
        prob_r = softmax_layer(h_r, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class, 'r')


    r_y = tf.reverse(y, [False, True])
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss = - tf.reduce_mean(y * tf.log(prob_o)) - tf.reduce_mean((tf.ones(tf.shape(y)) - y) * tf.log(prob_r)) + sum(reg_loss)
    # prob = FLAGS.alpha * prob_o + (1.0 - FLAGS.alpha) * (tf.ones(tf.shape(prob_r)) - prob_r)
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

        tr_x, tr_y, tr_sen_len, tr_doc_len = load_inputs_document(
            FLAGS.train_file_path,
            word_id_mapping_o,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        te_x, te_y, te_sen_len, te_doc_len = load_inputs_document(
            FLAGS.test_file_path,
            word_id_mapping_o,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        tr_x_r, tr_y_r, tr_sen_len_r, tr_doc_len_r = load_inputs_document(
            FLAGS.train_file_path_r,
            word_id_mapping_r,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        te_x_r, te_y_r, te_sen_len_r, te_doc_len_r = load_inputs_document(
            FLAGS.test_file_path_r,
            word_id_mapping_r,
            FLAGS.max_sentence_len,
            FLAGS.max_doc_len
        )
        # v_x, v_y, v_sen_len, v_doc_len = load_inputs_document(
        #     FLAGS.validate_file_path,
        #     word_id_mapping,
        #     FLAGS.max_sentence_len,
        #     FLAGS.max_doc_len
        # )

        # v_x, v_y, v_sen_len, v_doc_len = load_inputs_document(
        #     FLAGS.validate_file_path,
        #     word_id_mapping,
        #     FLAGS.max_sentence_len,
        #     FLAGS.max_doc_len
        # )

        def get_batch_data(xo, slo, dlo, xr, slr, dlr, yy, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yy), batch_size, 1, is_shuffle):
                feed_dict = {
                    x_o: xo[index],
                    x_r: xr[index],
                    y: yy[index],
                    sen_len_o: slo[index],
                    sen_len_r: slr[index],
                    doc_len_o: dlo[index],
                    doc_len_r: dlr[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc, max_prob, step = 0., None, None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, tr_x_r, tr_sen_len_r, tr_doc_len_r, tr_y,
                                           FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)

            saver.save(sess, save_dir, global_step=step)

            acc, cost, cnt, p = 0., 0., 0, []
            for test, num in get_batch_data(te_x, te_sen_len, te_doc_len, te_x_r, te_sen_len_r, te_doc_len_r, te_y, 2000, 1.0, 1.0, False):
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


if __name__ == '__main__':
    tf.app.run()
