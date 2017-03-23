#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
sys.path.append(os.getcwd())

from sklearn.metrics import precision_score, recall_score, f1_score
from newbie_nn.config import *
from newbie_nn.nn_layer import dynamic_rnn, bi_dynamic_rnn, softmax_layer
from newbie_nn.att_layer import bilinear_attention_layer
from data_prepare.utils import load_w2v, batch_index, load_sentence


def ian(inputs_l, len_l, inputs_r, len_r, keep_prob1, keep_prob2, _id='1'):
    cell = tf.nn.rnn_cell.LSTMCell
    # left hidden
    inputs_l = tf.nn.dropout(inputs_l, keep_prob=keep_prob1)
    hiddens_l = dynamic_rnn(cell, inputs_l, FLAGS.n_hidden, len_l, FLAGS.max_sentence_len, 'l' + _id, 'all')
    pool_l = tf.reduce_mean(hiddens_l, 1, keep_dims=False)
    # right hidden
    inputs_r = tf.nn.dropout(inputs_r, keep_prob=keep_prob1)
    hiddens_r = dynamic_rnn(cell, inputs_r, FLAGS.n_hidden, len_r, FLAGS.max_sentence_len, 'r' + _id, 'all')
    pool_r = tf.reduce_mean(hiddens_r, 1, keep_dims=False)
    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_r, len_l, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'l')
    outputs_l = tf.squeeze(tf.batch_matmul(att_l, hiddens_l))
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_l, len_r, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'r')
    outputs_r = tf.squeeze(tf.batch_matmul(att_r, hiddens_r))

    outputs = tf.concat(1, [outputs_l, outputs_r])
    prob = softmax_layer(outputs, 2 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)
    return prob


def main(_):
    word_id_mapping, w2v = load_w2v(FLAGS.embedding_file_path, FLAGS.embedding_dim, True)
    word_embedding = tf.constant(w2v, dtype=tf.float32)

    with tf.name_scope('inputs'):
        keep_prob1 = tf.placeholder(tf.float32)
        keep_prob2 = tf.placeholder(tf.float32)
        x_l = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        x_r = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        len_l = tf.placeholder(tf.int32, [None])
        len_r = tf.placeholder(tf.int32, [None])
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])

    inputs_l = tf.nn.embedding_lookup(word_embedding, x_l)
    inputs_r = tf.nn.embedding_lookup(word_embedding, x_r)
    prob = ian(inputs_l, len_l, inputs_r, len_r, keep_prob1, keep_prob2)

    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)

    acc_num, acc_prob = acc_func(y, prob)
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)
    optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.5).minimize(loss, global_step=global_step)
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

        tr_x1, tr_x2, tr_len1, tr_len2, tr_y = load_sentence(
            FLAGS.train_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len
        )

        te_x1, te_x2, te_len1, te_len2, te_y = load_sentence(
            FLAGS.test_file_path,
            word_id_mapping,
            FLAGS.max_sentence_len
        )

        def get_batch_data(x1, x2, len1, len2, yy, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yy), batch_size, 1, is_shuffle):
                feed_dict = {
                    x_l: x1[index],
                    x_r: x2[index],
                    y: yy[index],
                    len_l: len1[index],
                    len_r: len2[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc, max_prob, step = 0., None, None
        max_ty, max_py = None, None
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x1, tr_x2, tr_len1, tr_len2, tr_y,  FLAGS.batch_size,
                                           FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)
                # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                # sess.run(embed_update)

            acc, cost, cnt = 0., 0., 0
            p, ty, py = [], [], []
            for test, num in get_batch_data(te_x1, te_x2, te_len1, te_len2, te_y, 2000, 1.0, 1.0, False):
                _loss, _acc, _p, _ty, _py = sess.run([loss, acc_num, prob, true_y, pred_y], feed_dict=test)
                p += list(_p)
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
                max_prob = p
                max_ty = ty
                max_py = py
                saver.save(sess, save_dir, global_step=step)

        print 'P:', precision_score(max_ty, max_py, average=None)
        print 'R:', recall_score(max_ty, max_py, average=None)
        print 'F:', f1_score(max_ty, max_py, average=None)

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
