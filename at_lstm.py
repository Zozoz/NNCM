#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import tensorflow as tf
from nn_layer import dynamic_rnn, softmax_layer
from att_layer import dot_produce_attention_layer
from config import *
from utils import load_w2v, batch_index, load_word_embedding, load_aspect2id, load_inputs_twitter_at


def AE(inputs, target, length, keep_prob1, keep_prob2, type_='last'):
    print 'I am AE.'
    batch_size = tf.shape(inputs)[0]
    target = tf.reshape(target, [-1, 1, FLAGS.embedding_dim])
    target = tf.ones([batch_size, FLAGS.max_sentence_len, FLAGS.embedding_dim], dtype=tf.float32) * target
    inputs = tf.concat(2, [inputs, target])
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)

    cell = tf.nn.rnn_cell.LSTMCell
    outputs = dynamic_rnn(cell, inputs, FLAGS.n_hidden, length, FLAGS.max_sentence_len, 'AE', type_)
    return softmax_layer(outputs, FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)


def AT(inputs, target, length, keep_prob1, keep_prob2, type_='all'):
    print 'I am AT.'
    batch_size = tf.shape(inputs)[0]
    target = tf.reshape(target, [-1, 1, FLAGS.embedding_dim])
    target = tf.ones([batch_size, FLAGS.max_sentence_len, FLAGS.embedding_dim], dtype=tf.float32) * target
    in_t = tf.concat(2, [inputs, target])
    in_t = tf.nn.dropout(in_t, keep_prob=keep_prob1)
    cell = tf.nn.rnn_cell.LSTMCell
    hiddens = dynamic_rnn(cell, in_t, FLAGS.n_hidden, length, FLAGS.max_sentence_len, 'AT', 'all')
    h_t = tf.concat(2, [hiddens, target])

    alpha = dot_produce_attention_layer(h_t, length, FLAGS.n_hidden + FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 1)

    r = tf.reshape(tf.batch_matmul(alpha, hiddens), [-1, FLAGS.n_hidden])
    index = tf.range(0, batch_size) * FLAGS.max_sentence_len + (length - 1)
    hn = tf.gather(tf.reshape(hiddens, [-1, FLAGS.n_hidden]), index)  # batch_size * n_hidden

    Wp = tf.get_variable(
        name='Wp',
        shape=[FLAGS.n_hidden, FLAGS.n_hidden],
        initializer=tf.random_uniform_initializer(-0.01, 0.01),
        regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)
    )
    Wx = tf.get_variable(
        name='Wx',
        shape=[FLAGS.n_hidden, FLAGS.n_hidden],
        initializer=tf.random_uniform_initializer(-0.01, 0.01),
        regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg)
    )
    h = tf.tanh(tf.matmul(r, Wp) + tf.matmul(hn, Wx))
    return softmax_layer(h, FLAGS.n_hidden, FLAGS.random_base, keep_prob2, FLAGS.l2_reg, FLAGS.n_class)


def main(_):
    word_id_mapping, w2v = load_word_embedding(FLAGS.word_id_file_path, FLAGS.embedding_file_path, FLAGS.embedding_dim)
    word_embedding = tf.Variable(w2v, dtype=tf.float32, name='word_embedding')
    aspect_id_mapping, aspect_embed = load_aspect2id(FLAGS.aspect_id_file_path, word_id_mapping, w2v, FLAGS.embedding_dim)
    aspect_embedding = tf.Variable(aspect_embed, dtype=tf.float32, name='aspect_embedding')

    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x')
        y = tf.placeholder(tf.int32, [None, FLAGS.n_class], name='y')
        sen_len = tf.placeholder(tf.int32, None, name='sen_len')
        aspect_id = tf.placeholder(tf.int32, None, name='aspect_id')

    inputs = tf.nn.embedding_lookup(word_embedding, x)
    aspect = tf.nn.embedding_lookup(aspect_embedding, aspect_id)
    if FLAGS.method == 'AE':
        prob = AE(inputs, aspect, sen_len, keep_prob1, keep_prob2, FLAGS.t1)
    elif FLAGS.method == 'AT':
        prob = AT(inputs, aspect, sen_len, keep_prob1, keep_prob2, FLAGS.t1)

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
        train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer = summary_func(loss, acc_prob, _dir, title, sess)

        save_dir = 'model/' + str(timestamp) + '_' + title + '/'
        saver = saver_func(save_dir)

        init = tf.initialize_all_variables()
        sess.run(init)

        # saver.restore(sess, '/-')

        tr_x, tr_sen_len, tr_target_word, tr_y = load_inputs_twitter_at(
            FLAGS.train_file_path,
            word_id_mapping,
            aspect_id_mapping,
            FLAGS.max_sentence_len,
        )
        te_x, te_sen_len, te_target_word, te_y = load_inputs_twitter_at(
            FLAGS.test_file_path,
            word_id_mapping,
            aspect_id_mapping,
            FLAGS.max_sentence_len,
        )

        def get_batch_data(xi, sen_leni, yi, target_words, batch_size, kp1, kp2, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: xi[index],
                    y: yi[index],
                    sen_len: sen_leni[index],
                    aspect_id: target_words[index],
                    keep_prob1: kp1,
                    keep_prob2: kp2,
                }
                yield feed_dict, len(index)

        max_acc = 0.
        for i in xrange(FLAGS.n_iter):
            for train, _ in get_batch_data(tr_x, tr_sen_len, tr_y, tr_target_word, FLAGS.batch_size,
                                                FLAGS.keep_prob1, FLAGS.keep_prob2):
                _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                train_summary_writer.add_summary(summary, step)

            acc, cost, cnt = 0., 0., 0
            flag = True
            summary, step = None, None
            for test, num in get_batch_data(te_x, te_sen_len, te_y, te_target_word, 2000, 1.0, 1.0, False):
                _loss, _acc, _summary, _step = sess.run(
                    [loss, acc_num, validate_summary_op, global_step],
                    feed_dict=test)
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
