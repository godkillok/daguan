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

embed_size=100
batch_size=32
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

def

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


