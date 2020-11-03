#! /user/bin/evn python3
# -*- coding:utf8 -*-

"""
@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@copyright: "Copyright (c) 2020 Guoxiu He. All Rights Reserved"
"""

import os, sys, time, pickle, logging, codecs, json, argparse
import numpy as np
import tensorflow as tf
import random
tf.set_random_seed(1234)

sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../../')

from TensorFlow.Network import Network
import math, resource


class SIRM(object):
    def __init__(self, maxlen=150, nb_classes=2, nb_words=2000,
                 embedding_dim=200, dense_dim=200, rnn_dim=200, cnn_filters=200,
                 dropout_rate=0.5, learning_rate=0.001, weight_decay=0.0, l2_reg_lambda=0.0, optim_type='adam',
                 gpu='0', memory=0, data_struc='general', **kwargs):

        self.logger = logging.getLogger("Tensorflow")

        # data config
        self.nb_classes = nb_classes
        self.nb_words = nb_words

        # network config
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.rnn_dim = rnn_dim
        self.cnn_filters = cnn_filters
        self.dropout_rate = dropout_rate

        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.initializer = tf.truncated_normal_initializer(stddev=0.02)

        # optimizer config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.l2_reg_lambda = l2_reg_lambda
        self.optim_type = optim_type

        # session info config
        self.gpu = gpu
        self.memory = memory
        self.data_struc = data_struc

        if self.memory > 0:
            num_threads = os.environ.get('OMP_NUM_THREADS')
            self.logger.info("Memory use is %s." % self.memory)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=float(self.memory))
            config = tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads)
            self.sess = tf.Session(config=config)
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

        self.maxlen = 40
        self.sentmaxlen = 40

        self.num_hidden_layers = 6
        self.hidden_size = 200
        self.num_heads = 8
        self.filter_size = 200
        self.mode = 'avg'
        self.transfer_af = tf.nn.relu
        self.cnn_af = tf.nn.relu
        self.mlp_af = tf.nn.relu
        self.transfer_bias = False

        self.low_rate = 0.001

        self.kernel_sizes = [1, 2, 3]

        self.model_name = 'SIRM'
        self.data_name = 'tmp'


    def set_from_model_config(self, model_config):

        self.embedding_dim = model_config['embedding_dim']
        self.rnn_dim = model_config['rnn_dim']
        self.dense_dim = model_config['dense_dim']
        self.cnn_filters = model_config['cnn_filters']
        self.dropout_rate = model_config['dropout_rate']
        self.optim_type = model_config['optimizer']
        self.learning_rate = model_config['learning_rate']
        self.weight_decay = model_config['weight_decay']
        self.low_rate = model_config['low_rate']

        self.num_hidden_layers = model_config['num_hidden_layers']
        self.hidden_size = model_config['hidden_size']
        self.num_heads = model_config['num_heads']
        self.filter_size = model_config['filter_size']
        self.kernel_sizes = model_config['kernel_sizes']

        self.logger.info("set from model_config.")


    def set_from_data_config(self, data_config):
        self.nb_classes = data_config['nb_classes']
        self.logger.info("set from data_config.")


    def _setup_placeholders(self):
        self.input_x = tf.placeholder(tf.int32, [None, None, None], name="input_x")
        self.input_mask_x = tf.cast(tf.cast(self.input_x, tf.bool), tf.float32, name="input_mask_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.nb_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.logger.info("setup placeholders.")


    def _embed(self, embedding_matrix=np.array([None])):
        with tf.variable_scope('word_embedding'):
            self.embedding_softmax_layer = EmbeddingSharedWeights(self.nb_words, self.embedding_dim)
            self.logger.info("get embed.")


    def _inference(self):
        (self.batch_size, self.sentmaxlen, self.maxlen) = tf.unstack(tf.shape(self.input_x))
        self.input_x_reshape = tf.reshape(self.input_x, shape=[-1, self.maxlen])
        show_layer_info_with_memory('input_x_reshape', self.input_x_reshape, self.logger)

        self.embedded = self.embedding_softmax_layer(self.input_x_reshape)
        show_layer_info_with_memory('embedded', self.embedded, self.logger)

        with tf.name_scope("add_pos_encoding"):
            self.pos_encoding = get_position_encoding_matrix(self.maxlen, self.embedding_dim)
            show_layer_info_with_memory('pos_encoding', self.pos_encoding, self.logger)

            self.encoder_inputs = self.embedded + self.pos_encoding
            show_layer_info_with_memory('encoder_inputs_with_pos', self.encoder_inputs, self.logger)

        self.encoder_inputs = tf.layers.batch_normalization(self.encoder_inputs)
        show_layer_info_with_memory('bn_encoder_inputs', self.encoder_inputs, self.logger)

        if self.dropout_rate:
            self.encoder_inputs = tf.nn.dropout(self.encoder_inputs, self.dropout_keep_prob)
            show_layer_info_with_memory('do_encoder_inputs', self.encoder_inputs, self.logger)

        self.skim_information = self._skim_read()
        show_layer_info_with_memory('skim_information', self.skim_information, self.logger)

        with tf.variable_scope('word'):

            with tf.variable_scope('near_neighbor'):
                self.word_nn_output = tf.keras.layers.Conv1D(filters=self.hidden_size,
                                                             kernel_size=3,
                                                             padding='same',
                                                             # activation='relu',
                                                             kernel_initializer=self.initializer,
                                                             name='word_nn'
                                                             )(self.encoder_inputs)
                show_layer_info_with_memory('word_nn_output', self.word_nn_output, self.logger)

                self.word_nn_output = self.cnn_af(tf.layers.batch_normalization(self.word_nn_output))
                show_layer_info_with_memory('bn_word_nn_output', self.word_nn_output, self.logger)

                if self.dropout_rate:
                    self.word_nn_output = tf.nn.dropout(self.word_nn_output, self.dropout_keep_prob)
                    show_layer_info_with_memory('do_word_nn_output', self.word_nn_output, self.logger)

            with tf.variable_scope('skim'):
                self.word_skim_information_expand = tf.expand_dims(self.skim_information, axis=1)
                show_layer_info_with_memory('skim_information_expand', self.word_skim_information_expand, self.logger)
                self.word_skim_information = tf.tile(self.word_skim_information_expand,
                                                     multiples=[self.sentmaxlen, self.maxlen, 1])
                show_layer_info_with_memory('word_skim_information', self.word_skim_information, self.logger)

            with tf.variable_scope('dense_connect'):

                self.word_pre_hidden_tmp = tf.concat([self.encoder_inputs,
                                                      self.word_nn_output,
                                                      self.word_skim_information],
                                                     axis=-1)
                show_layer_info_with_memory('word_pre_hidden_tmp', self.word_pre_hidden_tmp, self.logger)

                self.word_outputs = tf.layers.dense(self.word_pre_hidden_tmp,
                                                       self.hidden_size,
                                                       use_bias=True,
                                                       # activation=tf.nn.tanh,
                                                       kernel_initializer=self.initializer,
                                                       name='word_pre_hidden')
                show_layer_info_with_memory('word_pre_hidden', self.word_outputs, self.logger)

                self.word_outputs = tf.nn.relu(tf.layers.batch_normalization(self.word_outputs))
                show_layer_info_with_memory('bn_word_pre_hidden', self.word_outputs, self.logger)

                if self.dropout_rate:
                    self.word_outputs = tf.nn.dropout(self.word_outputs, self.dropout_keep_prob)
                    show_layer_info_with_memory('do_word_pre_hidden', self.word_outputs, self.logger)

            with tf.variable_scope('avg_pooling'):
                self.word_level_mask = tf.reshape(self.input_mask_x, [-1, self.maxlen])
                self.word_level_output = tf.reduce_sum(tf.multiply(self.word_outputs,
                                                                   tf.nn.softmax(tf.expand_dims(self.word_level_mask,
                                                                                                axis=-1),
                                                                                 axis=-1)),
                                                       axis=1)

        self.sentence_inputs = tf.reshape(self.word_level_output, [-1, self.sentmaxlen, self.hidden_size])
        self.sentence_inputs_mask = tf.cast(tf.cast(tf.reduce_sum(self.input_mask_x, -1), tf.bool), tf.float32)
        show_layer_info_with_memory('sentence_inputs', self.sentence_inputs, self.logger)

        with tf.name_scope("add_pos_encoding"):
            self.sent_pos_encoding = get_position_encoding_matrix(self.sentmaxlen, self.hidden_size)
            show_layer_info_with_memory('sent_pos_encoding', self.sent_pos_encoding, self.logger)

            self.sentence_inputs = self.sentence_inputs + self.sent_pos_encoding
            show_layer_info_with_memory('sentence_inputs_with_pos', self.sentence_inputs, self.logger)

        self.sentence_inputs = tf.layers.batch_normalization(self.sentence_inputs)
        show_layer_info_with_memory('bn_sentence_inputs', self.sentence_inputs, self.logger)

        if self.dropout_rate:
            self.sentence_inputs = tf.nn.dropout(self.sentence_inputs, keep_prob=self.dropout_keep_prob)
            show_layer_info_with_memory('do_sentence_inputs', self.sentence_inputs, self.logger)

        with tf.variable_scope('sentence'):

            with tf.variable_scope('skim'):
                self.sentence_skim_information_expand = tf.expand_dims(self.skim_information, axis=1)
                show_layer_info_with_memory(
                    'sentence_skim_information_expand', self.sentence_skim_information_expand, self.logger)
                self.sentence_skim_information = tf.tile(self.sentence_skim_information_expand,
                                                         multiples=[1, self.sentmaxlen, 1])
                show_layer_info_with_memory(
                    'sentence_skim_information', self.sentence_skim_information, self.logger)

            with tf.variable_scope('near_neighbor'):
                self.sentence_nn_output = tf.keras.layers.Conv1D(filters=self.hidden_size,
                                                                 kernel_size=3,
                                                                 padding='same',
                                                                 # activation='relu',
                                                                 kernel_initializer=self.initializer,
                                                                 name='sentence_nn'
                                                                 )(self.sentence_inputs)
                show_layer_info_with_memory('sentence_nn_output', self.sentence_nn_output, self.logger)

                self.sentence_nn_output = self.cnn_af(tf.layers.batch_normalization(self.sentence_nn_output))
                show_layer_info_with_memory('bn_sentence_nn_output', self.sentence_nn_output, self.logger)

                if self.dropout_rate:
                    self.sentence_nn_output = tf.nn.dropout(self.sentence_nn_output, self.dropout_keep_prob)
                    show_layer_info_with_memory('do_sentence_nn_output', self.sentence_nn_output, self.logger)

            with tf.variable_scope('dense_connect'):

                self.sentence_pre_hidden_tmp = tf.concat([self.sentence_inputs,
                                                          self.sentence_nn_output,
                                                          self.sentence_skim_information],
                                                         axis=-1)
                show_layer_info_with_memory('sentence_pre_hidden_tmp', self.sentence_pre_hidden_tmp, self.logger)

                self.sentence_outputs = tf.layers.dense(self.sentence_pre_hidden_tmp,
                                                           self.hidden_size,
                                                           use_bias=True,
                                                           # activation=tf.tanh,
                                                           kernel_initializer=self.initializer,
                                                           name='sentence_pre_hidden')
                show_layer_info_with_memory('sentence_pre_hidden', self.sentence_outputs, self.logger)

                self.sentence_outputs = tf.nn.relu(tf.layers.batch_normalization(self.sentence_outputs))
                show_layer_info_with_memory('bn_sentence_pre_hidden', self.sentence_outputs, self.logger)

                if self.dropout_rate:
                    self.sentence_outputs = tf.nn.dropout(self.sentence_outputs, self.dropout_keep_prob)
                    show_layer_info_with_memory('do_sentence_pre_hidden', self.sentence_outputs, self.logger)

            with tf.variable_scope('avg_pooling'):
                self.sentence_level_output = tf.reduce_sum(tf.multiply(self.sentence_outputs,
                                                                       tf.nn.softmax(tf.expand_dims(self.sentence_inputs_mask,
                                                                                                    axis=-1),
                                                                                     axis=-1)),
                                                           axis=1)

        self.final_feature = tf.concat([self.skim_information, self.sentence_level_output], axis=-1)
        show_layer_info_with_memory('final_feature', self.final_feature, self.logger)

        #######################################################################################

        with tf.variable_scope("output"):

            self.logits = tf.keras.layers.Dense(self.nb_classes)(self.final_feature)
            show_layer_info_with_memory('Logits', self.logits, self.logger)

            self.output = tf.argmax(self.logits, axis=-1)
            show_layer_info_with_memory('output', self.output, self.logger)

            self.proba = tf.nn.top_k(tf.nn.softmax(self.logits)).values
            show_layer_info_with_memory('proba', self.proba, self.logger)

        with tf.variable_scope("low"):
            self.low_feature = FlipGradientBuilder()(self.skim_information)
            show_layer_info_with_memory('low_feature', self.low_feature, self.logger)
            low_feature_shape = self.low_feature.get_shape().as_list()
            W_projection = tf.get_variable("W_projection",
                                           shape=[low_feature_shape[-1], self.nb_classes],
                                           initializer=self.initializer)
            b_projection = tf.get_variable("b_projection", shape=[self.nb_classes])

            self.low = tf.matmul(self.low_feature, W_projection) + b_projection
            show_layer_info_with_memory('low', self.low, self.logger)

        self.logger.info("SkimIntensiveReading4 inference.")

        return self.logits


    def _compute_loss(self):

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,
                                                       logits=self.logits))
        self.logger.info("Calculate Loss.")

        self.low_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,
                                                       logits=self.low))
        self.logger.info("Calculate Low Loss.")

        self.loss = self.loss + self.low_rate * self.low_loss
        self.logger.info("Aggregate Loss.")

        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss
            self.logger.info("Add L2 Loss.")


    def _skim_read(self):
        self.skim_input = tf.reshape(self.embedded, shape=[-1, self.maxlen*self.sentmaxlen, self.embedding_dim])
        self.skim_input_mask = tf.reshape(self.input_mask_x, shape=[-1, self.maxlen*self.sentmaxlen])
        show_layer_info_with_memory('skim_input', self.skim_input, self.logger)

        self.skim_input_expanded = self.skim_input
        show_layer_info_with_memory('skim_input_expanded', self.skim_input_expanded, self.logger)

        pooled_outputs = []
        for i, kernel_size in enumerate(self.kernel_sizes):
            with tf.variable_scope("convolution-pooling-%s" % kernel_size):

                conv_b = self.cnn_af(tf.layers.batch_normalization(tf.keras.layers.Conv1D(self.cnn_filters,
                                                                                          kernel_size,
                                                                                          padding='SAME')
                                                                   (self.skim_input_expanded)), "relu")
                conv_b = self.cnn_af(tf.layers.batch_normalization(tf.keras.layers.Conv1D(self.cnn_filters,
                                                                                          kernel_size,
                                                                                          padding='SAME')
                                                                   (conv_b)), "relu")
                show_layer_info_with_memory('bn_convolutional-%s' % kernel_size, conv_b, self.logger)

                self.skim_input_mask_softmax = tf.nn.softmax(self.skim_input_mask, axis=-1)
                pooled = tf.reduce_sum(tf.multiply(conv_b,
                                                   tf.expand_dims(self.skim_input_mask_softmax, axis=-1)),
                                       axis=1, name='pool')
                show_layer_info_with_memory('avg pooling-%s' % kernel_size, pooled, self.logger)
                if self.dropout_rate:
                    pooled = tf.layers.dropout(pooled, self.dropout_keep_prob)
                    show_layer_info_with_memory('avg pooling-%s' % kernel_size, pooled, self.logger)

                pooled_outputs.append(pooled)

        sentence_skim = tf.concat(pooled_outputs, -1, name='concat_transfer')
        show_layer_info_with_memory('sentence_skim', sentence_skim, self.logger)

        return sentence_skim


    def train(self, data_generator, dropout_rate, epochs, folder_name, data_name,
              mode='train', batch_size=20, nb_classes=2, shuffle=True,
              is_val=True, is_test=True, save_best=True, monitor='f1score',
              val_mode='test_0', test_mode='test_0'):

        max_val = 0

        for epoch in range(epochs):
            self.logger.info('Training the model for epoch {} with batch size {}'.format(epoch, batch_size))
            start_t = int(time.time())

            counter, total_loss = 0, 0.0
            total_accuracy, total_precision, total_recall, total_f1score, total_macro_f1score, total_f0_5score, total_f2score = \
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            for X, Y in data_generator(folder_name=folder_name, data_name=data_name, mode=mode,
                                       batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle):
                feed_dict = {self.input_x: X,
                             self.input_y: Y,
                             self.dropout_keep_prob: dropout_rate}
                _, step, loss, logits = self.sess.run([self.train_op, self.global_step,
                                                       self.loss, self.logits], feed_dict)
                if epoch == 0 and counter == 0:
                    self.logger.info("X:\n%s"%X[0])
                    self.logger.info("Y:\n%s"%Y[0])
                    self.logger.info("input_mask_x:\n{}".format(self.sess.run(self.input_mask_x, feed_dict)[0]))
                    self.logger.info("sentence_inputs_mask:\n{}".format(self.sess.run(self.sentence_inputs_mask,
                                                                                      feed_dict)[0]))
                accuracy, precision, recall, f1score, macro_f1score, f0_5score, f2score = \
                    self._get_a_p_r_f_sara(input_y=Y, prediction=logits, category=1)
                total_loss += loss
                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1score += f1score
                total_macro_f1score += macro_f1score
                total_f0_5score += f0_5score
                total_f2score += f2score
                counter += 1

            self.logger.info(
                "%s: Epoch %d Batch %d Train Loss:%.4f\tAcc:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1Score:%.4f\tMacro_F1Score:%.4f\tF0_5Score:%.4f\tF2Score:%.4f"
                % (mode, epoch, counter, total_loss/float(counter), total_accuracy/float(counter),
                   total_precision/float(counter), total_recall/float(counter),
                   total_f1score/float(counter),
                   total_macro_f1score/float(counter),
                   total_f0_5score/float(counter),
                   total_f2score/float(counter),
                   )
            )
            self.logger.info('Epoch time: %sh: %sm: %ss'%(int((int(time.time()) - start_t) / 3600),
                                                          int((int(time.time()) - start_t) % 3600 / 60),
                                                          int((int(time.time()) - start_t) % 3600 % 60)
                                                          ))
            if is_val:
                val_t = time.time()
                loss, accuracy, precision, recall, f1score, macro_f1score, f0_5score, f2score = \
                    self.evaluate_batch(data_generator, folder_name, data_name, mode=val_mode,
                                        batch_size=batch_size, nb_classes=nb_classes, shuffle=False)
                self.logger.info('Evaluate time: %sh: %sm: %ss' % (int((int(time.time()) - val_t) / 3600),
                                                                   int((int(time.time()) - val_t) % 3600 / 60),
                                                                   int((int(time.time()) - val_t) % 3600 % 60)
                                                                   ))
                metrics_dict = {"loss": loss, "accuracy": accuracy,
                                "precision": precision, "recall": recall,
                                "f1score": f1score, "macro_f1score": macro_f1score, "f0_5score": f0_5score, "f2score": f2score}
                if metrics_dict[monitor] > max_val and save_best:
                    max_val = metrics_dict[monitor]
                    self.save(self.save_dir, self.model_name)

        self.save(self.save_dir, self.model_name+'.last')

        if is_test:
            test_t = time.time()
            self.evaluate_batch(data_generator, folder_name, data_name, mode=test_mode,
                                batch_size=batch_size, nb_classes=nb_classes, shuffle=False)
            self.logger.info('Test time: %sh: %sm: %ss' % (int((int(time.time()) - test_t) / 3600),
                                                           int((int(time.time()) - test_t) % 3600 / 60),
                                                           int((int(time.time()) - test_t) % 3600 % 60)
                                                           ))


    def evaluate_batch(self, data_generator, folder_name, data_name, mode='test_0',
                       batch_size=20, nb_classes=2, shuffle=False):
        counter, total_loss = 0, 0.0
        total_accuracy, total_precision, total_recall, total_f1score, total_macro_f1score, total_f0_5score, total_f2score = \
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for X, Y in data_generator(folder_name=folder_name, data_name=data_name, mode=mode,
                                   batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle):
            feed_dict = {self.input_x: X,
                         self.input_y: Y,
                         self.dropout_keep_prob: 1.0}

            loss, logits = self.sess.run([self.loss, self.logits], feed_dict)

            accuracy, precision, recall, f1score, macro_f1score, f0_5score, f2score = \
                self._get_a_p_r_f_sara(input_y=Y, prediction=logits, category=1)
            total_loss += loss
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1score += f1score
            total_macro_f1score += macro_f1score
            total_f0_5score += f0_5score
            total_f2score += f2score
            counter += 1

        self.logger.info(
            "%s: Batch %d Loss:%.4f\tAcc:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1Score:%.4f\tMacro F1Score:%.4f\tF0_5Score:%.4f\tF2Score:%.4f"
            % (mode, counter,
               total_loss / float(counter), total_accuracy / float(counter),
               total_precision / float(counter), total_recall / float(counter),
               total_f1score / float(counter),
               total_macro_f1score / float(counter),
               total_f0_5score / float(counter),
               total_f2score / float(counter),
               )
        )

        return total_loss/float(counter), total_accuracy/float(counter),\
               total_precision/float(counter), total_recall/float(counter), total_f1score/float(counter), \
               total_macro_f1score / float(counter), total_f0_5score/float(counter), total_f2score/float(counter)


class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size, method="gather"):
        """Specify characteristic parameters of embedding layer.

        Args:
            vocab_size: Number of tokens in the embedding. (Typically ~32,000)
            hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
            method: Strategy for performing embedding lookup. "gather" uses tf.gather
                which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
                one-hot encodes the indicies and formulates the embedding as a sparse
                matrix multiplication. The matmul formulation is wasteful as it does
                extra work, however matrix multiplication is very fast on TPUs which
                makes "matmul" considerably faster than "gather" on TPUs.
        """
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        if method not in ("gather", "matmul"):
            raise ValueError("method {} must be 'gather' or 'matmul'".format(method))
        self.method = method

    def build(self, _):
        with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
            # Create and initialize weights. The random normal initializer was chosen
            # randomly, and works well.
            self.shared_weights = tf.get_variable(
                "weights", [self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(0., self.hidden_size ** -0.5)
            )
        self.built = True

    def call(self, x):
        """Get token embeddings of x.

        Args:
            x: An int64 tensor with shape [batch_size, length]
        Returns:
            embeddings: float32 tensor with shape [batch_size, length, embedding_size]
            padding: float32 tensor with shape [batch_size, length] indicating the locations of the padding tokens in x.
        """
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            mask = tf.to_float(tf.not_equal(x, 0))

            embeddings = tf.gather(self.shared_weights, x)
            embeddings *= tf.expand_dims(mask, -1)
            # embedding_matmul already zeros out masked positions, so
            # `embeddings *= tf.expand_dims(mask, -1)` is unnecessary.

            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

            return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.

        Args:
            x: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
            float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


def get_position_encoding_matrix(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
        length: Sequence length.
        hidden_size: Size of the
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position

    Returns:
        Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def show_memory_use():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom = rusage_denom * rusage_denom
    ru = resource.getrusage(resource.RUSAGE_SELF)
    total_memory = 1. * (ru.ru_maxrss + ru.ru_ixrss +
                         ru.ru_idrss + ru.ru_isrss) / rusage_denom
    strinfo = "\x1b[33m [Memory] Total Memory Use: %.4f MB \t Resident: %ld Shared: %ld UnshareData: %ld UnshareStack: %ld \x1b[0m" % \
              (total_memory, ru.ru_maxrss, ru.ru_ixrss, ru.ru_idrss, ru.ru_isrss)
    return strinfo


def show_layer_info_with_memory(layer_name, layer_out, logger=None):
    if logger:
        logger.info('[layer]: %s\t[shape]: %s \n%s'
                    % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))
    else:
        print('[layer]: %s\t[shape]: %s \n%s'
              % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))


if __name__ == '__main__':
    logger = logging.getLogger("SIRM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' ')[:3])

    network = SIRM()

    log_path = './logs/' + network.model_name + '.log'
    if os.path.exists(log_path):
        os.remove(log_path)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    network.build_graph()
