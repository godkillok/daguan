from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import os
import json
from nltk import tokenize
import wordsegment
# for python 2.x
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir3", "Base directory for the model.")
flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate")
flags.DEFINE_float("learning_rate", 0.4, "Learning rate")
flags.DEFINE_integer("embedding_size", 128, "embedding size")
flags.DEFINE_integer("hidden_size", 120, "hidden size")
flags.DEFINE_integer("atten_size", 110, "atten_size")
flags.DEFINE_integer("num_filters", 100, "number of filters")
flags.DEFINE_integer("num_classes", 14, "number of classes")
flags.DEFINE_integer("shuffle_buffer_size", 20000, "dataset shuffle buffer size")
flags.DEFINE_integer("sentence_max_len", 60, "max length of sentences")
flags.DEFINE_integer("batch_size", 128, "number of instances in a batch")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_string("data_dir", "/home/tom/new_data/daguan/text/dbpedia_csv/", "Directory containing the dataset")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated list of number of window size in each filter")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
FLAGS = flags.FLAGS
import nltk

nltk.download('punkt')
dropout_keep_proba=0.8
MAX_SENT_LENGTH = 60
MAX_SENTS = 3
MAX_NB_WORDS = 20000


def parse_line(line, vocab):
    def get_content(record):
        fields = record.decode().split(",")
        if len(fields) < 3:
            raise ValueError("invalid record %s" % record)
        sentences = tokenize.sent_tokenize(fields[2])
        sen_list = []
        word_len = []
        sen_len = 0
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                text = re.sub(r"[^A-Za-z0-9\'\`]", " ", sent)
                text = re.sub(r"\s{2,}", " ", text)
                text = re.sub(r"\`", "\'", text)
                text = text.strip().lower()
                tokens = text.split()
                tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0]
                n = len(tokens)
                if n >= FLAGS.sentence_max_len:
                    tokens = tokens[:FLAGS.sentence_max_len]
                    word_len.append(len(tokens))
                if n < FLAGS.sentence_max_len:
                    tokens += [FLAGS.pad_word] * (FLAGS.sentence_max_len - n)
                    word_len.append(n)

                sen_list+=(tokens)
        sen_num = len(word_len)
        word_num=len(sen_list)
        if len(word_len) < MAX_SENTS:
            for i in range(MAX_SENTS -sen_num):
                sen_list+=([FLAGS.pad_word] * (FLAGS.sentence_max_len))
                word_len.append(0)
        # print('sence {}'.format(len(sen_list)))
        if len(sen_list)>180:
            print(len(sen_list))
            sen_list=sen_list[:180]
        return [sen_list, np.int32(fields[0]), np.int32(sen_num), np.int32(word_num)]

    result = tf.py_func(get_content, [line], [tf.string, tf.int32, tf.int32, tf.int32])
    result[0].set_shape([MAX_SENTS*FLAGS.sentence_max_len])
    result[1].set_shape([])
    result[2].set_shape([])
    result[3].set_shape([])
    # Lookup tokens to return their ids
    ids = vocab.lookup(result[0])
    return {"sentence": ids, 'word_len': result[3], 'sen_len': result[2]}, result[1] - 1


def input_fn(path_csv, path_vocab, shuffle_buffer_size, num_oov_buckets):
    """Create tf.data Instance from csv file
    Args:
        path_csv: (string) path containing one example per line
        vocab: (tf.lookuptable)
    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens and labels for each example
    """
    vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=num_oov_buckets)
    # Load txt file, one example per line
    dataset = tf.data.TextLineDataset(path_csv)
    # Convert line into list of tokens, splitting by white space
    dataset = dataset.map(lambda line: parse_line(line, vocab))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size).repeat()
    dataset = dataset.batch(FLAGS.batch_size).prefetch(1)
    print(dataset.output_types)
    print(dataset.output_shapes)
    return dataset


def attention_word_level(hidden_state,context_vecotor_word,scope):
    """
    input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
    input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
    :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
    """

    hidden_size=FLAGS.hidden_size
    atten_size=FLAGS.atten_size
    hidden_state_ = tf.concat(hidden_state, axis=2)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
    hidden_state_=tf.reshape(hidden_state_,[-1,MAX_SENT_LENGTH,2*hidden_size])
    # 0) one layer of feed forward network
    hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,hidden_size * 2])  # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
    # hidden_state_:[batch_size*num_sentences*sequence_length,hidden_size*2];W_w_attention_sentence:[,hidden_size*2,,hidden_size*2]

    hidden_representation=tf.layers.dense(hidden_state_2,atten_size* 3,use_bias=True,activation=tf.nn.tanh)
    # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
    hidden_representation = tf.reshape(hidden_representation, shape=[-1, MAX_SENT_LENGTH,
                                                                     atten_size * 3])  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
    # attention process:1.get logits for each word in the sentence. 2.get possibility distribution for each word in the sentence. 3.get weighted sum for the sentence as sentence representation.
    # 1) get logits for each word in the sentence.
    hidden_state_context_similiarity = tf.multiply(hidden_representation, context_vecotor_word)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
    attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                     axis=2)  # shape:[batch_size*num_sentences,sequence_length]
    # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
    attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                         keep_dims=True)  # shape:[batch_size*num_sentences,1]
    # 2) get possibility distribution for each word in the sentence.
    p_attention = tf.nn.softmax(
        attention_logits - attention_logits_max)  # shape:[batch_size*num_sentences,sequence_length]
    # 3) get weighted hidden state by attention vector,this is word attention, since word=batch_size*num_sentences*sequence_length

    p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[batch_size*num_sentences,sequence_length,1]
    # below sentence_representation'shape:[batch_size*num_sentences,sequence_length,hidden_size*2]<----p_attention_expanded:[batch_size*num_sentences,sequence_length,1];hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
    #multiply  element-wise, so p_attention_expanded  [batch_size*num_sentences,sequence_length,1]
    #hidden_state_ [batch_size*num_sentences,sequence_length,hidden_size*2]
    #p_attention_expanded*hidden_state_=[batch_size*num_sentences,sequence_length,hidden_size*2]
    sentence_representation = tf.multiply(p_attention_expanded,
                                          hidden_state_)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
    #this sum is to calcualte per word  hidden tensor after attention weight
    sentence_representation = tf.reduce_sum(sentence_representation,
                                            axis=1)  # shape:[batch_size*num_sentences,hidden_size*2]
    # because per word has already multiply their weight, sentence_representation will reduce num of word in sentence, only left batch_size*num_sentences
    return sentence_representation  # shape:[batch_size*num_sentences,hidden_size*2]



def attention_sentence_level(hidden_state,context_vecotor_sentence,scope):
    """
    input1:self.hidden_state: hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
    input2:sentence level context vector:[batch_size*num_sentences,hidden_size*2]
    :return:representation.shape:[batch_size*num_sentences,hidden_size*2]
    """

    hidden_size=FLAGS.hidden_size
    atten_size=FLAGS.atten_size
    hidden_state_ = tf.concat(hidden_state, axis=2)  # shape:[batch_size,num_sentences,hidden_size*2]

    # 0) one layer of feed forward network
    hidden_state_2 = tf.reshape(hidden_state_, shape=[-1,MAX_SENTS,hidden_size * 4])  # shape:[batch_size*num_sentences,hidden_size*4]
    # hidden_state_:[batch_size*num_sentences*sequence_length,hidden_size*2];W_w_attention_sentence:[,hidden_size*2,,hidden_size*2]

    hidden_representation=tf.layers.dense(hidden_state_2,atten_size* 2,use_bias=True,activation=tf.nn.tanh)
    # shape:[batch_size*num_sentences*sequence_length,hidden_size*2]
    hidden_representation = tf.reshape(hidden_representation, shape=[-1,MAX_SENTS,
                                                                     atten_size * 2])  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
    # attention process:1.get logits for each word in the sentence. 2.get possibility distribution for each word in the sentence. 3.get weighted sum for the sentence as sentence representation.
    # 1) get logits for each word in the sentence.
    hidden_state_context_similiarity = tf.multiply(hidden_representation, context_vecotor_sentence)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
    attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                     axis=2)  # shape:[batch_size*num_sentences,sequence_length]
    # subtract max for numerical stability (softmax is shift invariant). tf.reduce_max:Computes the maximum of elements across dimensions of a tensor.
    attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                         keep_dims=True)  # shape:[1]
    # 2) get possibility distribution for each word in the sentence.
    p_attention = tf.nn.softmax(
        attention_logits - attention_logits_max)  # shape:shape:[None]
    # 3) get weighted hidden state by attention vector
    p_attention_expanded = tf.expand_dims(p_attention, axis=2)  # shape:[batch_size,1]
    # below sentence_representation'shape:[batch_size*num_sentences,sequence_length,hidden_size*2]<----p_attention_expanded:[batch_size*num_sentences,sequence_length,1];hidden_state_:[batch_size*num_sentences,sequence_length,hidden_size*2]
    sentence_representation = tf.multiply(p_attention_expanded,
                                          hidden_state_)  # shape:[batch_size,sequence_length,hidden_size*2]
    sentence_representation = tf.reduce_sum(sentence_representation,
                                            axis=1)  # shape:[batch_size,hidden_size*2]
    return sentence_representation  #  shape:[None,hidden_size*2]


def my_model(features, labels, mode, params):
    sentence = features['sentence']
    word_len = features['word_len']
    sen_len = features['sen_len']
    # Get word embeddings for each token in the sentence
    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[params["vocab_size"], FLAGS.embedding_size])
    context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[FLAGS.atten_size * 3],
                                           dtype=tf.float32)  # TODO o.k to use batch_size in first demension?

    context_vecotor_sentence = tf.get_variable("what_is_the_informative_sentence",
                                               shape=[FLAGS.atten_size * 2], dtype=tf.float32)

    sentence = tf.nn.embedding_lookup(embeddings,
                                      sentence)  # shape:(batch, num of sentence,sentence_len, embedding_size)
    # add a channel dim, required by the conv2d and max_pooling2d method
    # sentence = tf.expand_dims(sentence, -1) # shape:(batch, sentence_len/height, embedding_size/width, channels=1)

    pooled_outputs = []
    word_level_inputs = tf.reshape(sentence, [-1, MAX_SENTS * FLAGS.sentence_max_len, FLAGS.embedding_size])
    word_level_lengths = word_len

    # lstm模型　正方向传播的RNN
    word_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size, forget_bias=1.0)
    # 反方向传播的RNN
    word_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size, forget_bias=1.0)

    sentence_fw_cell=tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size*2, forget_bias=1.0)

    sentence_bw_cell=tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size*2, forget_bias=1.0)

    with tf.variable_scope('word') as scope:
        word_encoder_output, _ = tf.nn.bidirectional_dynamic_rnn(
            word_fw_cell, word_bw_cell,
            word_level_inputs, word_level_lengths,
            scope=scope, dtype=tf.float32)
        #
        word_representation = attention_word_level(
            word_encoder_output, context_vecotor_word,
            scope=scope)
        #
        # word_encoder_output=   tf.concat(word_encoder_output, axis=2)  # shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        # hidden_state_=tf.reshape(word_encoder_output,[-1,2*FLAGS.hidden_size])

        word_representation = tf.layers.dropout(
            word_representation, rate=(1-dropout_keep_proba) ) # shape:[batch_size*num_sentences,hidden_size*2]

    sentence_inputs = tf.reshape(
        word_representation, [-1, MAX_SENTS, FLAGS.hidden_size*2])  # shape:[batch_sizes,num_sentence,hidden_size*2]

    with tf.variable_scope('sentence') as scope:
        sentence_encoder_output, _ = tf.nn.bidirectional_dynamic_rnn(
            sentence_fw_cell, sentence_bw_cell,
            sentence_inputs, sen_len,
            scope=scope, dtype=tf.float32)

        sentence_representation = attention_sentence_level(
            sentence_encoder_output, context_vecotor_sentence,
            scope=scope)
        # sentence_representation=shape:[batch_sizes,hidden_size*2]
        sentence_representation = tf.layers.dropout(
            sentence_representation, rate=(1-dropout_keep_proba) )


    with tf.name_scope("output"):
        logits = tf.layers.dense(sentence_representation,FLAGS.num_classes) # shape:[None,self.num_classes]==tf.matmul([None,hidden_size*2],[hidden_size*2,self.num_classes])

    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    def _train_op_fn(loss):
        return optimizer.minimize(loss, global_step=tf.train.get_global_step())

    my_head = tf.contrib.estimator.multi_class_head(n_classes=FLAGS.num_classes)
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=_train_op_fn
    )


def main(unused_argv):
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    # Loads parameters from json file
    with open(json_path) as f:
        config = json.load(f)
    FLAGS.pad_word = config["pad_word"]
    if config["train_size"] < FLAGS.shuffle_buffer_size:
        FLAGS.shuffle_buffer_size = config["train_size"]
    print("shuffle_buffer_size:", FLAGS.shuffle_buffer_size)

    # Get paths for vocabularies and dataset
    path_words = os.path.join(FLAGS.data_dir, 'words.txt')
    assert os.path.isfile(path_words), "No vocab file found at {}, run build_vocab.py first".format(path_words)
    # words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=config["num_oov_buckets"])

    path_train = os.path.join(FLAGS.data_dir, 'train_sh.csv')
    path_eval = os.path.join(FLAGS.data_dir, 'test_shuf.csv')

    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'vocab_size': config["vocab_size"],
            'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate
        },
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(path_train, path_words, FLAGS.shuffle_buffer_size, config["num_oov_buckets"]),
        max_steps=FLAGS.train_steps
    )
    input_fn_for_eval = lambda: input_fn(path_eval, path_words, 0, config["num_oov_buckets"])
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("after train and evaluate")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)


