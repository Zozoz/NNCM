#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from newbie_nn.nn_layer import dynamic_rnn, softmax_layer, bi_dynamic_rnn
from newbie_nn.att_layer import dot_produce_attention_layer, bilinear_attention_layer, mlp_attention_layer
from newbie_nn.config import *
from data_prepare.utils import load_w2v, batch_index, load_word_embedding, load_aspect2id, load_inputs_twitter


def TD_att(input_fw, input_bw, sen_len_fw, sen_len_bw, target, keep_prob1, keep_prob2, type_='last'):
    print 'I am TD-ATT.'
    cell = tf.nn.rnn_cell.LSTMCell
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    # forward
    hidden_fw = dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'TC-ATT-1', type_)
    ht_fw = tf.concat(2, [hidden_fw, target])
    # alpha_fw = dot_produce_attention_layer(ht_fw, sen_len_fw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
    alpha_fw = mlp_attention_layer(ht_fw, sen_len_fw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
    # alpha_fw = bilinear_attention_layer(hidden_fw, target, sen_len_fw, FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
    r_fw = tf.reshape(tf.batch_matmul(alpha_fw, hidden_fw), [-1, FLAGS.n_hidden])

    # backward
    hidden_bw = dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'TC-ATT-2', type_)
    ht_bw = tf.concat(2, [hidden_bw, target])
    # alpha_bw = dot_produce_attention_layer(ht_bw, sen_len_bw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
    alpha_bw = mlp_attention_layer(ht_bw, sen_len_bw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
    # alpha_bw = bilinear_attention_layer(hidden_bw, target, sen_len_bw, FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
    r_bw = tf.reshape(tf.batch_matmul(alpha_bw, hidden_bw), [-1, FLAGS.n_hidden])

    output = tf.concat(1, [r_fw, r_bw])
    return softmax_layer(output, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class), alpha_fw, alpha_bw


def TD(input_fw, input_bw, sen_len_fw, sen_len_bw, target, keep_prob1, keep_prob2, type_='last'):
    print 'I am TD.'
    cell = tf.nn.rnn_cell.LSTMCell
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    # forward
    hn_fw = dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'TC-ATT-1', type_)

    # backward
    hn_bw = dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'TC-ATT-2', type_)

    output = tf.concat(1, [hn_fw, hn_bw])
    return softmax_layer(output, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)


def TD_bi(input_fw, input_bw, sen_len_fw, sen_len_bw, target, keep_prob1, keep_prob2, type_='last'):
    print 'I am TD-BI.'
    cell = tf.nn.rnn_cell.LSTMCell
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    # forward
    hn_fw = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'TC-ATT-1', type_)

    # backward
    hn_bw = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'TC-ATT-2', type_)

    output = tf.concat(1, [hn_fw, hn_bw])
    return softmax_layer(output, 4 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim)
    word_embedding = tf.constant(w2v, name='word_embedding')
    # word_embedding = tf.Variable(w2v, name='word_embedding')

    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
        sen_len = tf.placeholder(tf.int32, None)

        x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        sen_len_bw = tf.placeholder(tf.int32, [None])

        target_words = tf.placeholder(tf.int32, [None, 1])

    inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
    inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
    target = tf.nn.embedding_lookup(word_embedding, target_words)
    # for MLP & DOT
    batch_size = tf.shape(inputs_bw)[0]
    target = tf.zeros([batch_size, FLAGS.max_sentence_len, FLAGS.embedding_dim]) + target
    # for BL
    # target = tf.squeeze(target)
    alpha_fw, alpha_bw = None, None
    if FLAGS.method == 'TD-ATT':
        prob, alpha_fw, alpha_bw = TD_att(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, keep_prob1, keep_prob2, 'all')
    elif FLAGS.method == 'TD-BI':
        prob = TD_bi(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, keep_prob1, keep_prob2, FLAGS.t1)
    else:
        prob = TD(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, keep_prob1, keep_prob2, FLAGS.t1)

    loss = loss_func(y, prob)
    acc_num, acc_prob = acc_func(y, prob)
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = train_func(loss, FLAGS.learning_rate, global_step)
    true_y = tf.argmax(y, 1)
    pred_y = tf.argmax(prob, 1)

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
        train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer = summary_func(loss, acc_prob, _dir, title, sess)

        save_dir = 'model/' + str(timestamp) + '_' + title + '/'
        saver = saver_func(save_dir)

        init = tf.initialize_all_variables()
        sess.run(init)

        # saver.restore(sess, '/-')

        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word = load_inputs_twitter(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC'
        )
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word = load_inputs_twitter(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC'
        )

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_fw, max_bw = None, None
        max_ty, max_py = None, None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, FLAGS.batch_size,
                                                FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)

            acc, cost, cnt = 0., 0., 0
            flag = True
            summary, step = None, None
            fw, bw, ty, py = [], [], [], []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, 2000, 1.0, 1.0, False):
                _loss, _acc, _summary, _step, _fw, _bw, _ty, _py = sess.run(
                    [loss, acc_num, validate_summary_op, global_step, alpha_fw, alpha_bw, true_y, pred_y],
                    feed_dict=test)
                fw += list(_fw)
                bw += list(_bw)
                ty += list(_ty)
                py += list(_py)
                acc += _acc
                cost += _loss * num
                cnt += num
                if flag:
                    summary = _summary
                    step = _step
                    flag = False
            print 'all samples={}, correct prediction={}'.format(cnt, acc)
            test_summary_writer.add_summary(summary, step)
            saver.save(sess, save_dir, global_step=step)
            print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, cost / cnt, acc / cnt)
            if acc / cnt > max_acc:
                max_acc = acc / cnt
                max_fw = fw
                max_bw = bw
                max_ty = ty
                max_py = py
        print 'P:', precision_score(max_ty, max_py, average=None)
        print 'R:', recall_score(max_ty, max_py, average=None)
        print 'F:', f1_score(max_ty, max_py, average=None)

        fp = open(FLAGS.prob_file + '_fw', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_fw):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')
        fp = open(FLAGS.prob_file + '_bw', 'w')
        for y1, y2, ws in zip(max_ty, max_py, max_bw):
            fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws[0]]) + '\n')

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
