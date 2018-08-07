#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#test
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
from tflearn.data_utils import to_categorical, pad_sequences

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
                      "./data/embeding/wide_deep_test.csv.tfrecords",
                      "Validate files which supports glob pattern")
  flags.DEFINE_string("inference_data_file", "./data/embeding/wide_deep_test.csv",
                      "Data file for inference")
  flags.DEFINE_string("inference_result_file", "./inference_result.txt",
                      "Result file from inference")
  flags.DEFINE_string("optimizer", "adagrad",
                      "Support sgd, adadelta, adagrad, adam, ftrl, rmsprop")
  flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
  flags.DEFINE_string("model", "lr",
                      "Support dnn, lr, wide_and_deep, customized, cnn")
  flags.DEFINE_string("dnn_struct", "128 32 8", "DNN struct")
  flags.DEFINE_integer("epoch_number", 100, "Number of epoches")
  flags.DEFINE_integer("batch_size", 2, "Batch size")
  flags.DEFINE_integer("validate_batch_size", 1024,
                       "Batch size for validation")
  flags.DEFINE_integer("batch_thread_number", 1, "Batch thread number")
  flags.DEFINE_integer("min_after_dequeue", 100, "Min after dequeue")
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

def assert_flags(FLAGS):
  if FLAGS.mode in ["train", "inference", "savedmodel"]:
    if FLAGS.scenario in ["classification", "regression"]:
      if FLAGS.train_file_format in ["tfrecords", "csv"]:
        if FLAGS.optimizer in [
            "sgd", "adadelta", "adagrad", "adam", "ftrl", "rmsprop"
        ]:
          if FLAGS.model in [
              "dnn", "lr", "wide_and_deep", "customized", "cnn"
          ]:
            return

  logging.error("Get the unsupported parameters, exit now")
  exit(1)

def restore_from_checkpoint(sess, saver, checkpoint):
  if checkpoint:
    logging.info("Restore session from checkpoint: {}".format(checkpoint))
    saver.restore(sess, checkpoint)
    return True
  else:
    logging.warn("Checkpoint not found: {}".format(checkpoint))
    return False


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
        batch_size=FLAGS.batch_size,
        dynamic_pad=True)
    # item2vec = tf.nn.embedding_lookup_sparse(aa, batch_features['education_num'], None, combiner="sum")
    label = tf.feature_column.input_layer(batch_features, label_columns)
    deep_features = tf.feature_column.input_layer(batch_features, deep_columns)
    return label, deep_features

def full_connect(inputs, weights_shape, biases_shape, is_train=True):
  weights = tf.get_variable(
      "weights", weights_shape, initializer=tf.random_normal_initializer())
  biases = tf.get_variable(
      "biases", biases_shape, initializer=tf.random_normal_initializer())
  layer = tf.matmul(inputs, weights) + biases

  if FLAGS.enable_bn and is_train:
    mean, var = tf.nn.moments(layer, axes=[0])
    scale = tf.get_variable(
        "scale", biases_shape, initializer=tf.random_normal_initializer())
    shift = tf.get_variable(
        "shift", biases_shape, initializer=tf.random_normal_initializer())
    layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                      FLAGS.bn_epsilon)
  return layer

def inference(batch_features, input_units, output_units, is_train):
    with tf.variable_scope("lr"):
        layer = full_connect(batch_features, [input_units, output_units], [output_units])
    return layer


logging.basicConfig(level=logging.INFO)
FLAGS = define_flags()
assert_flags(FLAGS)
pprint.PrettyPrinter().pprint(FLAGS.__flags)
if FLAGS.enable_colored_log:
  import coloredlogs
  coloredlogs.install()

def main():
    # Get hyper-parameters
    if os.path.exists(FLAGS.checkpoint_path) == False:
        os.makedirs(FLAGS.checkpoint_path)
    CHECKPOINT_FILE = FLAGS.checkpoint_path + "/checkpoint.ckpt"
    LATEST_CHECKPOINT = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    if os.path.exists(FLAGS.output_path) == False:
        os.makedirs(FLAGS.output_path)

    EPOCH_NUMBER = FLAGS.epoch_number
    if EPOCH_NUMBER <= 0:
        EPOCH_NUMBER = None

    BATCH_CAPACITY = FLAGS.batch_thread_number * FLAGS.batch_size + FLAGS.min_after_dequeue


    read_and_decode_function = read_and_decode_tfrecords


    train_filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(FLAGS.train_file), num_epochs=EPOCH_NUMBER)
    batch_labels, batch_features = read_and_decode_function(train_filename_queue)
    batch_features = pad_sequences(batch_features, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    # testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length

    # batch_labels, batch_features = tf.train.shuffle_batch(
    #     [train_label, train_features],
    #     batch_size=FLAGS.batch_size,
    #     num_threads=FLAGS.batch_thread_number,
    #     capacity=BATCH_CAPACITY,
    #     min_after_dequeue=FLAGS.min_after_dequeue)

    validate_filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(FLAGS.validate_file),
        num_epochs=EPOCH_NUMBER)

    validate_batch_labels, validate_batch_features = read_and_decode_function(
        validate_filename_queue)

    # validate_label, validate_features = read_and_decode_function(
    #     validate_filename_queue)
    # validate_batch_labels, validate_batch_features = tf.train.shuffle_batch(
    #     [validate_label, validate_features],
    #     batch_size=FLAGS.validate_batch_size,
    #     num_threads=FLAGS.batch_thread_number,
    #     capacity=BATCH_CAPACITY,
    #     min_after_dequeue=FLAGS.min_after_dequeue)

    # Define the model
    # input_units = FLAGS.feature_size
    # output_units = FLAGS.label_size
    input_units = batch_features.shape[1]
    output_units = FLAGS.label_size
    logging.info("Use the model: {}, model network: {}".format(
        FLAGS.model, FLAGS.dnn_struct))

    logits = inference(batch_features, input_units, output_units, True)


    batch_labels = tf.to_int64(batch_labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=batch_labels)
    loss = tf.reduce_mean(cross_entropy, name="loss")
    l2_lambda = 0.0001
    l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
    loss = loss + l2_losses

    global_step = tf.Variable(0, name="global_step", trainable=False)
    if FLAGS.enable_lr_decay:
        logging.info(
            "Enable learning rate decay rate: {}".format(FLAGS.lr_decay_rate))
        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(
            starter_learning_rate,
            global_step,
            100000,
            FLAGS.lr_decay_rate,
            staircase=True)
    else:
        learning_rate = FLAGS.learning_rate
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.get_variable_scope().reuse_variables()



    # Define accuracy op for train data
    train_accuracy_logits = inference(batch_features, input_units, output_units,
                                      False)
    train_softmax = tf.nn.softmax(train_accuracy_logits)
    train_correct_prediction = tf.equal(
        tf.argmax(train_softmax, 1), batch_labels)
    train_accuracy = tf.reduce_mean(
        tf.cast(train_correct_prediction, tf.float32))

    # Define auc op for train data
    # 这段代码1.将batch_label变成int32
    batch_labels = tf.cast(batch_labels, tf.int32)
    # 这段代码2.将batch_label变成一列数据
    sparse_labels = tf.reshape(batch_labels, [-1, 1])
    # 算算有多少个样本
    derived_size = tf.shape(batch_labels)[0]
    # 对样本从0开始编号
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    # 将样本编号合并和batch_label，得到每个样本，预测得到label
    concated = tf.concat(axis=1, values=[indices, sparse_labels])

    # one hot 的结果是样本行，预测label可能取值的size列的数据
    outshape = tf.stack([derived_size, FLAGS.label_size])

    # mainly from the sparse to dense in  one hot way
    new_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    _, train_auc = tf.contrib.metrics.streaming_auc(train_softmax,
                                                    new_batch_labels)

    # Define accuracy op for validate data
    validate_accuracy_logits = inference(validate_batch_features, input_units,
                                         output_units, False)
    validate_softmax = tf.nn.softmax(validate_accuracy_logits)
    validate_batch_labels = tf.to_int64(validate_batch_labels)
    validate_correct_prediction = tf.equal(
        tf.argmax(validate_softmax, 1), validate_batch_labels)
    validate_accuracy = tf.reduce_mean(
        tf.cast(validate_correct_prediction, tf.float32))

    # Define auc op for validate data
    validate_batch_labels = tf.cast(validate_batch_labels, tf.int32)
    sparse_labels = tf.reshape(validate_batch_labels, [-1, 1])
    derived_size = tf.shape(validate_batch_labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(axis=1, values=[indices, sparse_labels])
    outshape = tf.stack([derived_size, FLAGS.label_size])
    new_validate_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    _, validate_auc = tf.contrib.metrics.streaming_auc(validate_softmax,
                                                       new_validate_batch_labels)

    # Define inference op
    inference_features = tf.placeholder(
        "float", [None, input_units], name="features")
    inference_logits = inference(inference_features, input_units, output_units,
                                 False)
    inference_softmax = tf.nn.softmax(inference_logits, name="output_softmax")
    inference_op = tf.argmax(inference_softmax, 1, name="output_prediction")
    keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1], name="keys")
    keys_identity = tf.identity(keys_placeholder, name="output_keys")
    model_signature = signature_def_utils.build_signature_def(
        inputs={
            "keys": utils.build_tensor_info(keys_placeholder),
            "features": utils.build_tensor_info(inference_features)
        },
        outputs={
            "keys": utils.build_tensor_info(keys_identity),
            "prediction": utils.build_tensor_info(inference_op),
            "softmax": utils.build_tensor_info(inference_softmax),
        },
        method_name=signature_constants.PREDICT_METHOD_NAME)

    # Initialize saver and summary
    saver = tf.train.Saver()
    tf.summary.scalar("loss", loss)
    if FLAGS.scenario == "classification":
        tf.summary.scalar("train_accuracy", train_accuracy)
        tf.summary.scalar("train_auc", train_auc)
        tf.summary.scalar("validate_accuracy", validate_accuracy)
        tf.summary.scalar("validate_auc", validate_auc)
    summary_op = tf.summary.merge_all()
    init_op = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
        tf.tables_initializer()
    ]

    # Create session to run
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.output_path, sess.graph)
        sess.run(init_op)
        filter_sizes = [1, 2, 3, 4, 5]
        vocab_size=100
        textCN=textCNN(filter_sizes,FLAGS.num_filters,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate,FLAGS.sentence_len,vocab_size,FLAGS.embed_size,FLAGS.is_training)

        if FLAGS.mode == "train":
            # Restore session and start queue runner
            restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            start_time = datetime.datetime.now()

            try:
                while not coord.should_stop():
                    curr_loss, curr_acc, _ = sess.run([textCNN.train_op])
                    _, step, print_features = sess.run([train_op, global_step, batch_features])
                    np.set_printoptions(suppress=True)

                    # Print state while training
                    if step % FLAGS.steps_to_validate == 0:
                        if FLAGS.scenario == "classification":
                            loss_value, train_accuracy_value, train_auc_value, validate_accuracy_value, validate_auc_value, summary_value = sess.run(
                                [
                                    loss, train_accuracy, train_auc, validate_accuracy,
                                    validate_auc, summary_op
                                ])
                            end_time = datetime.datetime.now()
                            logging.info(
                                "[{}] Step: {}, loss: {}, train_acc: {}, train_auc: {}, valid_acc: {}, valid_auc: {}".
                                    format(end_time - start_time, step, loss_value,
                                           train_accuracy_value, train_auc_value,
                                           validate_accuracy_value, validate_auc_value))
                        elif FLAGS.scenario == "regression":
                            loss_value, summary_value = sess.run([loss, summary_op])
                            end_time = datetime.datetime.now()
                            logging.info("[{}] Step: {}, loss: {}".format(
                                end_time - start_time, step, loss_value))

                        writer.add_summary(summary_value, step)
                        saver.save(sess, CHECKPOINT_FILE, global_step=step)
                        # saver.save(sess, CHECKPOINT_FILE)
                        start_time = end_time
            except tf.errors.OutOfRangeError:
                if FLAGS.enable_benchmark:
                    print("Finish training for benchmark")
                    exit(0)
                else:
                    # Export the model after training
                    print("Do not export the model yet")

            finally:
                coord.request_stop()
            coord.join(threads)

        elif FLAGS.mode == "savedmodel":
            if restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT) == False:
                logging.error("No checkpoint for exporting model, exit now")
                exit(1)

            graph_file_name = "graph.pb"
            logging.info("Export the graph to: {}".format(FLAGS.model_path))
            tf.train.write_graph(
                sess.graph_def, FLAGS.model_path, graph_file_name, as_text=False)

            export_path = os.path.join(
                compat.as_bytes(FLAGS.model_path),
                compat.as_bytes(str(FLAGS.model_version)))
            logging.info("Export the model to {}".format(export_path))

            try:
                legacy_init_op = tf.group(
                    tf.tables_initializer(), name='legacy_init_op')
                builder = saved_model_builder.SavedModelBuilder(export_path)
                builder.add_meta_graph_and_variables(
                    sess, [tag_constants.SERVING],
                    clear_devices=True,
                    signature_def_map={
                        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            model_signature,
                    },
                    legacy_init_op=legacy_init_op)

                builder.save()
            except Exception as e:
                logging.error("Fail to export saved model, exception: {}".format(e))

        elif FLAGS.mode == "inference":
            if restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT) == False:
                logging.error("No checkpoint for inferencing, exit now")
                exit(1)

            # Load inference test data
            inference_result_file_name = FLAGS.inference_result_file
            inference_test_file_name = FLAGS.inference_data_file
            inference_data = np.genfromtxt(inference_test_file_name, delimiter=",")
            inference_data_features = inference_data[:, 0:9]
            inference_data_labels = inference_data[:, 9]

            # Run inference
            start_time = datetime.datetime.now()
            prediction, prediction_softmax = sess.run(
                [inference_op, inference_softmax],
                feed_dict={inference_features: inference_data_features})
            end_time = datetime.datetime.now()

            # Compute accuracy
            label_number = len(inference_data_labels)
            correct_label_number = 0
            for i in range(label_number):
                if inference_data_labels[i] == prediction[i]:
                    correct_label_number += 1
            accuracy = float(correct_label_number) / label_number

            # Compute auc
            y_true = np.array(inference_data_labels)
            y_score = prediction_softmax[:, 1]
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            logging.info("[{}] Inference accuracy: {}, auc: {}".format(
                end_time - start_time, accuracy, auc))

            # Save result into the file
            np.savetxt(inference_result_file_name, prediction_softmax, delimiter=",")
            logging.info(
                "Save result to file: {}".format(inference_result_file_name))


class textCNN():
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False,
                 clip_gradients=5.0, decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        # self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")  # y:[None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],
                                                 name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32)  # training iteration
        self.tst = tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.
        self.possibility = tf.nn.sigmoid(self.logits)
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.");self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.");self.loss_val = self.loss()
        self.train_op = self.train()
        if not self.multi_label_flag:
            self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
            print("self.predictions:", self.predictions)
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),
                                          self.input_y)  # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",
                                                shape=[self.num_classes])  # [label_size] #ADD 2017.06.09


