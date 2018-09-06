from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import os
import json
from tensorflow.contrib import rnn

# for python 22.x
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
# model_dir2 is the good one embedding size 128
flags = tf.app.flags
path='/home/tom/new_data/input_data'
# path='C:/Users/TangGuoping/Downloads'
flags.DEFINE_string("model_dir", "/media/tom/软件/model_dir3", "Base directory for the model.")
flags.DEFINE_string("train_file_pattern", "{}/*train.tfrecords".format(path), "train file pattern")
flags.DEFINE_string("eval_file_pattern", "{}/*eval.tfrecords".format(path), "evalue file pattern")
flags.DEFINE_string("pred_eval_file_pattern", "/home/tom/new_data/input_data/*teval.tfrecords", "evalue file pattern")
flags.DEFINE_string("pred_file_pattern", "/home/tom/new_data/input_data/*pred.tfrecords", "evalue file pattern")

flags.DEFINE_float("dropout_rate", 0.8, "Drop out rate")
flags.DEFINE_float("learning_rate", 0.8, "Learning rate")
flags.DEFINE_float("decay_rate", 0.7, "Learning rate")
flags.DEFINE_integer("embedding_size", 100, "embedding size")
flags.DEFINE_integer("num_filters", 100, "number of filters")
flags.DEFINE_integer("num_classes", 19, "number of classes")
flags.DEFINE_integer("num_parallel_readers", 4, "number of classes")
flags.DEFINE_integer("shuffle_buffer_size", 30000, "dataset shuffle buffer size")
flags.DEFINE_integer("sentence_max_len", 250, "max length of sentences")
flags.DEFINE_integer("batch_size", 128, "number of instances in a batch")
flags.DEFINE_integer("save_checkpoints_steps", 500, "Save checkpoints every this many steps")
flags.DEFINE_integer("train_steps", 20000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("decay_steps", 5000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("train_epoch", 1,
                     "Number of (global) training steps to perform")
flags.DEFINE_string("data_dir", "/home/tom/new_data/input_data/",
                    "Directory containing the dataset")
flags.DEFINE_string("test_dir", " /data/tanggp/deeplearning-master/word_cnn/dbpedia_csv/test*",
                    "Directory containing the dataset")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated list of number of window size in each filter")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
flags.DEFINE_string("path_vocab", "/home/tom/new_data/input_data/words.txt", "used for word index")
flags.DEFINE_string("fast_text", "/home/tom/new_data/super_more.bin", "used for word index")
FLAGS = flags.FLAGS


def parse_exmp(serialized_example):
    feats = tf.parse_single_example(
        serialized_example,
        features={
            "text": tf.FixedLenFeature([FLAGS.sentence_max_len], tf.int64),
            # "text": tf.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
            "label": tf.FixedLenFeature([], tf.int64)
        })

    labels = feats.pop('label')

    return feats, labels


def train_input_fn(filenames, shuffle_buffer_size, shuffle=True):
    # dataset = tf.data.TFRecordDataset(filenames) filename is a string
    print('tfrecord')
    print(filenames)
    files = tf.data.Dataset.list_files(filenames, shuffle=shuffle)  # A dataset of all files matching a pattern.
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
    dataset = dataset.map(parse_exmp, num_parallel_calls=10)

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat().batch(FLAGS.batch_size)

    print(dataset.output_types)
    print(dataset.output_shapes)
    return dataset


def assign_pretrained_word_embedding(params):
    print("using pre-trained word emebedding.started.word2vec_model_path:", FLAGS.fast_text)
    import fastText as ft
    word2vec_model = ft.load_model(FLAGS.fast_text)

    vocab_size = params["vocab_size"]
    word_embedding_final = np.zeros((vocab_size, FLAGS.embedding_size))  # create an empty word_embedding list.

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0

    with open(FLAGS.path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {l.strip(): i for i, l in enumerate(lines)}

    for (word, idx) in vocab.items():
        embedding = word2vec_model.get_word_vector(word)
        if embedding.max() == 'not a numeric object' or embedding.min() == 'not a numeric object':
            print('gg')
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_final[idx, :] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_final[idx, :] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.

    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor

    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")
    return word_embedding_final



def my_model(features, labels, mode, params):
    sentence = features['text']
    # Get word embeddings for each token in the sentence

    assert (params["vocab_size"], FLAGS.embedding_size) == params["word_embedding"].shape
    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[params["vocab_size"], FLAGS.embedding_size],
                                 initializer=tf.constant_initializer(params["word_embedding"], dtype=tf.float32))
    hidden_size=FLAGS.embedding_size
    sentence = tf.nn.embedding_lookup(embeddings, sentence)  # shape:(batch, sentence_len, embedding_size)
    # add a channel dim, required by the conv2d and max_pooling2d method


    lstm_fw_cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_size) #forward direction cell
    lstm_bw_cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_size) #backward direction cell
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
        lstm_fw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,output_keep_prob=params['dropout_rate'])
        lstm_bw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,output_keep_prob=params['dropout_rate'])

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, sentence,
                                                 dtype=tf.float32)  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
    print("outputs:===>",outputs)  # outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
    # 3. concat output
    output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
    output_rnn_last = tf.reduce_mean(output_rnn, axis=1)  # [batch_size,hidden_size*2] #output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2] #TODO
    print("output_rnn_last:", output_rnn_last)  # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
    # 4. logits(use linear layer)

    sentence2=tf.reduce_mean(sentence,axis=1)

    output_rnn_last1=tf.concat([output_rnn_last,sentence2], axis=1)
    logits = tf.layers.dense(output_rnn_last1, FLAGS.num_classes, activation=None)

    # learning_rate = tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(), FLAGS.decay_steps,
    #                                            FLAGS.decay_rate, staircase=True)
    learning_rate = params['learning_rate']
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

    def _train_op_fn(loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
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

    # path_train = os.path.join(FLAGS.data_dir, 'train.csv')
    # path_eval = os.path.join(FLAGS.data_dir, 'test.csv')
    path_train = FLAGS.train_file_pattern
    path_eval = FLAGS.eval_file_pattern
    params = {
        'vocab_size': config["vocab_size"],
        'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate
    }
    word_embedding = assign_pretrained_word_embedding(params)
    params['word_embedding'] = word_embedding
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, keep_checkpoint_max=2,save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    )

    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=lambda: input_fn(path_train, path_words, FLAGS.shuffle_buffer_size, config["num_oov_buckets"]),
    #     max_steps=FLAGS.train_steps
    # )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(path_train, FLAGS.shuffle_buffer_size),
        max_steps=FLAGS.train_steps
    )

    # input_fn_for_eval = lambda: input_fn(path_eval, path_words, 0, config["num_oov_buckets"])
    input_fn_for_eval = lambda: train_input_fn(path_eval, 0, shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, steps=60, throttle_secs=100)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    print("evalue train set")
    input_fn_for_pred = lambda: train_input_fn(path_train, 0)
    classifier.evaluate(input_fn=input_fn_for_pred, steps=120)

    input_fn_for_eval = lambda: train_input_fn(path_eval, 0)
    # eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, steps=30, throttle_secs=60)
    # tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    print("evalue eval set")
    classifier.evaluate(input_fn=input_fn_for_eval, steps=100)
    print("after train and evaluate")



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main=main)
    from cnn import nn_pred
    tf.app.run(main=nn_pred.pred(my_model,FLAGS,'rcnn1'))
