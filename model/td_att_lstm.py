#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from newbie_nn.nn_layer import dynamic_rnn, softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from newbie_nn.att_layer import dot_produce_attention_layer, bilinear_attention_layer, mlp_attention_layer, Mlp_attention_layer
from newbie_nn.config import *
from data_prepare.utils import load_w2v, batch_index, load_inputs_twitter
tf.app.flags.DEFINE_string('is_m', '1', 'prob')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.app.flags.DEFINE_string('is_bi', '1', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 10, 'max target length')


def ian_t(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, _id='all'):
    cell = tf.contrib.rnn.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    hiddens_l = dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    # right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = tf.reduce_mean(hiddens_t, 1, keep_dims=False)

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'l')
    outputs_l = tf.squeeze(tf.batch_matmul(att_l, hiddens_l))
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'r')
    outputs_r = tf.squeeze(tf.batch_matmul(att_r, hiddens_r))

    outputs = tf.concat([outputs_l, outputs_r], 1)
    prob = softmax_layer(outputs, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return prob


def ian(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, _id='all'):
    cell = tf.contrib.rnn.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    # pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)
    # right hidden
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    # pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)
    # target hidden
    target = tf.nn.dropout(target, keep_prob=keep_prob1)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)
    # pool_t = tf.concat(1, [target, target])

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'l')
    outputs_l = tf.squeeze(tf.batch_matmul(att_l, hiddens_l))
    # index_l = tf.argmax(tf.squeeze(att_l), 1)
    # pool_l = tf.gather(hiddens_l, index_l)
    # # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'r')
    outputs_r = tf.squeeze(tf.batch_matmul(att_r, hiddens_r))
    # index_r = tf.argmax(tf.squeeze(att_r), 1)
    # pool_r = tf.gather(hiddens_r, index_r)

    # attention target
    # att_t_l = bilinear_attention_layer(hiddens_t, pool_l, sen_len_tr, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'tl')
    # output_t_l = tf.squeeze(tf.batch_matmul(att_t_l, hiddens_t))
    # att_t_r = bilinear_attention_layer(hiddens_t, pool_r, sen_len_tr, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'tr')
    # output_t_r = tf.squeeze(tf.batch_matmul(att_t_r, hiddens_t))

    outputs = tf.concat([outputs_l, outputs_r, pool_t], 1)
    # outputs = tf.concat(1, [outputs_l, outputs_r, output_t_l, output_t_r])
    prob = softmax_layer(outputs, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return prob, att_l, att_r


def TD_att(input_fw, input_bw, sen_len_fw, sen_len_bw, target, keep_prob1, keep_prob2, type_='last'):
    print 'I am TD-ATT.'
    cell = tf.nn.rnn_cell.LSTMCell
    input_fw = tf.nn.dropout(input_fw, keep_prob=keep_prob1)
    input_bw = tf.nn.dropout(input_bw, keep_prob=keep_prob1)
    # forward
    if FLAGS.is_bi == '1':
        hidden_fw = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'TC-ATT-1', type_)
        ht_fw = tf.concat(2, [hidden_fw, target])
        # alpha_fw = dot_produce_attention_layer(ht_fw, sen_len_fw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
        alpha_fw = mlp_attention_layer(ht_fw, sen_len_fw, 2 * FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
        # alpha_fw = bilinear_attention_layer(hidden_fw, target, sen_len_fw, FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
        r_fw = tf.reshape(tf.batch_matmul(alpha_fw, hidden_fw), [-1, 2 * FLAGS.n_hidden])
    else:
        hidden_fw = dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'TC-ATT-1', type_)
        ht_fw = tf.concat(2, [hidden_fw, target])
        # alpha_fw = dot_produce_attention_layer(ht_fw, sen_len_fw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
        alpha_fw = mlp_attention_layer(ht_fw, sen_len_fw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
        # alpha_fw = bilinear_attention_layer(hidden_fw, target, sen_len_fw, FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)
        r_fw = tf.reshape(tf.batch_matmul(alpha_fw, hidden_fw), [-1, FLAGS.n_hidden])

    # backward
    if FLAGS.is_bi == '1':
        hidden_bw = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'TC-ATT-2', type_)
        ht_bw = tf.concat(2, [hidden_bw, target])
        # alpha_bw = dot_produce_attention_layer(ht_bw, sen_len_bw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
        alpha_bw = mlp_attention_layer(ht_bw, sen_len_bw, 2 * FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
        # alpha_bw = bilinear_attention_layer(hidden_bw, target, sen_len_bw, FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
        r_bw = tf.reshape(tf.batch_matmul(alpha_bw, hidden_bw), [-1, 2 * FLAGS.n_hidden])
    else:
        hidden_bw = dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'TC-ATT-2', type_)
        ht_bw = tf.concat(2, [hidden_bw, target])
        # alpha_bw = dot_produce_attention_layer(ht_bw, sen_len_bw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
        alpha_bw = mlp_attention_layer(ht_bw, sen_len_bw, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
        # alpha_bw = bilinear_attention_layer(hidden_bw, target, sen_len_bw, FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 2)
        r_bw = tf.reshape(tf.batch_matmul(alpha_bw, hidden_bw), [-1, FLAGS.n_hidden])

    output = tf.concat(1, [r_fw, r_bw])
    if FLAGS.is_bi == '1':
        return softmax_layer(output, 4 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class), alpha_fw, alpha_bw
    else:
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
    print_config()
    with tf.device('/gpu:1'):
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

            target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
            tar_len = tf.placeholder(tf.int32, [None])

        inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
        inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
        target = tf.nn.embedding_lookup(word_embedding, target_words)
        target = reduce_mean_with_len(target, tar_len)
        # for MLP & DOT
        target = tf.expand_dims(target, 1)
        batch_size = tf.shape(inputs_bw)[0]
        target = tf.zeros([batch_size, FLAGS.max_sentence_len, FLAGS.embedding_dim]) + target
        # for BL
        # target = tf.squeeze(target)
        alpha_fw, alpha_bw = None, None
        if FLAGS.method == 'IAN':
            prob = ian(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, keep_prob1, keep_prob2, 'all')
        elif FLAGS.method == 'TD-ATT':
            prob, alpha_fw, alpha_bw = TD_att(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, keep_prob1, keep_prob2, 'all')
        elif FLAGS.method == 'TD-BI':
            prob = TD_bi(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, keep_prob1, keep_prob2, FLAGS.t1)
        else:
            prob = TD(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, keep_prob1, keep_prob2, FLAGS.t1)

        loss = loss_func(y, prob)
        acc_num, acc_prob = acc_func(y, prob)
        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
        # optimizer = train_func(loss, FLAGS.learning_rate, global_step)
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

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
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

        if FLAGS.is_r == '1':
            is_r = True
        else:
            is_r = False

        tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len = load_inputs_twitter(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )
        te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len = load_inputs_twitter(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r,
            FLAGS.max_target_len
        )

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        max_fw, max_bw = None, None
        max_ty, max_py = None, None
        step = None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len,
                                           FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)
            saver.save(sess, save_dir, global_step=step)

            acc, cost, cnt = 0., 0., 0
            fw, bw, ty, py = [], [], [], []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                            te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                if FLAGS.method == 'TD-ATT':
                    _loss, _acc, _fw, _bw, _ty, _py = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, true_y, pred_y], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                else:
                    _loss, _acc, _ty, _py = sess.run([loss, acc_num, true_y, pred_y], feed_dict=test)
                ty += list(_ty)
                py += list(_py)
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
                max_fw = fw
                max_bw = bw
                max_ty = ty
                max_py = py
        P = precision_score(max_ty, max_py, average=None)
        R = recall_score(max_ty, max_py, average=None)
        F1 = f1_score(max_ty, max_py, average=None)
        print 'P:', P, 'avg=', sum(P) / FLAGS.n_class
        print 'R:', R, 'avg=', sum(R) / FLAGS.n_class
        print 'F1:', F1, 'avg=', sum(F1) / FLAGS.n_class

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
