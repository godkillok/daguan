#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import datetime
import logging
import os
import pprint

import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat
import  fastText
import FLAGS
embed_size=100
batch_size=32
EPOCH_NUMBER=2


def define_flags():
    flags = tf.app.flags
    flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log")
    flags.DEFINE_string("mode", "inference", "Support train, inference, savedmodel")
    flags.DEFINE_boolean("enable_benchmark", False, "Enable benchmark")
    flags.DEFINE_string("scenario", "classification",
                        "Support classification, regression")
    flags.DEFINE_integer("feature_size", 9, "Number of feature size")
    flags.DEFINE_integer("label_size", 2, "Number of label size")
    flags.DEFINE_string("train_file_format", "tfrecords",
                        "Support tfrecords, csv")
    flags.DEFINE_string("train_file", "./data/embeding/wide_deep_test.csv.tfrecords",
                        "Train files which supports glob pattern")
    flags.DEFINE_string("validate_file",
                        "./data/cancer/cancer_test.csv.tfrecords",
                        "Validate files which supports glob pattern")
    flags.DEFINE_string("inference_data_file", "./data/embeding/wide_deep_test.csv",
                        "Data file for inference")
    flags.DEFINE_string("inference_result_file", "./inference_result.txt",
                        "Result file from inference")
    flags.DEFINE_string("optimizer", "adagrad",
                        "Support sgd, adadelta, adagrad, adam, ftrl, rmsprop")
    flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
    flags.DEFINE_string("model", "dnn",
                        "Support dnn, lr, wide_and_deep, customized, cnn")
    flags.DEFINE_string("dnn_struct", "128 32 8", "DNN struct")
    flags.DEFINE_integer("epoch_number", 1, "Number of epoches")
    flags.DEFINE_integer("batch_size", 3, "Batch size")
    flags.DEFINE_integer("validate_batch_size", 1024,
                         "Batch size for validation")
    flags.DEFINE_integer("batch_thread_number", 1, "Batch thread number")
    flags.DEFINE_integer("min_after_dequeue", 10, "Min after dequeue")
    flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization")
    flags.DEFINE_float("bn_epsilon", 0.001, "Epsilon of batch normalization")
    flags.DEFINE_boolean("enable_dropout", False, "Enable dropout")
    flags.DEFINE_float("dropout_keep_prob", 0.5, "Keep prob of dropout")
    flags.DEFINE_boolean("enable_lr_decay", False, "Enable learning rate decay")
    flags.DEFINE_float("lr_decay_rate", 0.96, "Learning rate decay rate")
    flags.DEFINE_integer("steps_to_validate", 10, "Steps to validate")
    flags.DEFINE_string("checkpoint_path", "./checkpoint/",
                        "Path for checkpoint")
    flags.DEFINE_string("output_path", "./tensorboard/", "Path for tensorboard")
    flags.DEFINE_string("model_path", "./model/", "Path of the model")
    flags.DEFINE_integer("model_version", 1, "Version of the model")
    FLAGS = flags.FLAGS
    return FLAGS


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    text = tf.feature_column.numeric_column('text')

    deep_columns = [text]
    label = tf.feature_column.indicator_column('label')
    return  deep_columns, [label]

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN,word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    # word2vec_model = word2vec.load(word2vec_model_path)
    word2vec_model = fastText.load_model(word2vec_model_path)
    vocab=word2vec_model.get_words()
    vectors=[]
    for v in vocab:
        vectors.append(word2vec_model.get_word_vector(v))
    word2vec_dict = {}
    for word, vector in zip(vocab, vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

def read_and_decode_tfrecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    deep_columns,label_columns = build_model_columns()
    # embedding_initializer=tf.contrib.framework.load_embedding_initializer(
    #       ckpt_path='C:/work/tensorflow_template/log/model.ckpt')

    examples = tf.parse_single_example(
        serialized_example,
        features={
            "text": tf.VarLenFeature(tf.int64),
            "label": tf.VarLenFeature(tf.int64)
        })
    # batch_features = tf.train.shuffle_batch(
    #     examples,
    #     batch_size=FLAGS.batch_size,
    #     num_threads=FLAGS.batch_thread_number,
    #     capacity=16,
    #     min_after_dequeue=FLAGS.min_after_dequeue)
    batch_features = tf.train.batch(
        examples,
        batch_size=batch_size,
        dynamic_pad=True)
    # item2vec = tf.nn.embedding_lookup_sparse(aa, batch_features['education_num'], None, combiner="sum")
    label = tf.feature_column.input_layer(batch_features, label_columns)
    deep_features = tf.feature_column.input_layer(batch_features, deep_columns)
    return label, deep_features

def main():
    train_filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(FLAGS.train_file), num_epochs=EPOCH_NUMBER)
    train_label, train_features = read_and_decode_tfrecords(train_filename_queue)

class textCNN():
    def __int__(self):
        self.instantiate_weights()


    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",
                                                shape=[self.num_classes])  # [label_size] #ADD 2017.06.09


