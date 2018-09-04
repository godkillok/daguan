from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import os
import json

# for python 22.x1
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
#model_dir2 is the good one embedding size 128
flags = tf.app.flags
flags.DEFINE_string("model_dir", "/media/tom/软件/model_dir1", "Base directory for the model.")
flags.DEFINE_string("train_file_pattern", "/home/tom/new_data/input_data/*train.tfrecords", "train file pattern")
flags.DEFINE_string("eval_file_pattern", "/home/tom/new_data/input_data/*eval.tfrecords", "evalue file pattern")
flags.DEFINE_string("pred_eval_file_pattern", "/home/tom/new_data/input_data/*teval.tfrecords", "evalue file pattern")

flags.DEFINE_string("pred_file_pattern", "/home/tom/new_data/input_data/*pred.tfrecords", "evalue file pattern")

flags.DEFINE_float("dropout_rate", 0.5, "Drop out rate")
flags.DEFINE_float("learning_rate", 0.3, "Learning rate")
flags.DEFINE_float("decay_rate", 0.8, "L+earning rate")
flags.DEFINE_integer("embedding_size", 100, "embedding size")
flags.DEFINE_integer("num_filters", 200, "number of filters")
flags.DEFINE_integer("num_classes", 19, "number of classes")
flags.DEFINE_integer("num_parallel_readers", 4, "number of classes")
flags.DEFINE_integer("shuffle_buffer_size", 30000, "dataset shuffle buffer size")
flags.DEFINE_integer("sentence_max_len", 250, "max length of sentences")
flags.DEFINE_integer("batch_size", 128, "number of instances in a batch")
flags.DEFINE_integer("save_checkpoints_steps", 500, "Save checkpoints every this many steps")
flags.DEFINE_integer("train_steps", 40010,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("decay_steps", 5000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("train_epoch", 1,
                     "Number of (global) training steps to perform")
flags.DEFINE_string("data_dir", "/home/tom/new_data/input_data/",
                    "Directory containing the dataset")
flags.DEFINE_string("test_dir", " /data/tanggp/deeplearning-master/word_cnn/dbpedia_csv/test*",
                    "Directory containing the dataset")
flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated list of number of window size in each filter")
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

def parse_exmp2(serialized_example):
    feats = tf.parse_single_example(
        serialized_example,
        features={
            "text": tf.FixedLenFeature([FLAGS.sentence_max_len], tf.int64),
            "_id": tf.FixedLenFeature([], tf.int64),
            # "text": tf.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
            "label": tf.FixedLenFeature([], tf.int64)
        })

    labels = feats.pop('label')

    return feats, labels


def train_input_fn(filenames, shuffle_buffer_size,shuffle=True,repeat=0):
    # dataset = tf.data.TFRecordDataset(filenames) filename is a string
    print('train tfrecord')
    print(filenames)
    files = tf.data.Dataset.list_files(filenames, shuffle=shuffle)  # A dataset of all files matching a pattern.
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
    dataset = dataset.map(parse_exmp, num_parallel_calls=10)

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    if repeat>0:
        dataset = dataset.repeat(repeat).batch(FLAGS.batch_size)
    else:
        dataset = dataset.repeat().batch(FLAGS.batch_size)

    print(dataset.output_types)
    print(dataset.output_shapes)
    return dataset


def assign_pretrained_word_embedding(params):
    print("using pre-trained word emebedding.started.word2vec_model_path:",FLAGS.fast_text)
    import fastText as ft
    word2vec_model = ft.load_model(FLAGS.fast_text)

    vocab_size=params["vocab_size"]
    word_embedding_final = np.zeros((vocab_size,FLAGS.embedding_size))  # create an empty word_embedding list.

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0

    with open(FLAGS.path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {l.strip(): i for i, l in enumerate(lines)}

    for (word,idx) in vocab.items():
        embedding=word2vec_model.get_word_vector(word)
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_final[idx,:] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_final[idx, :] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.

    # word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor

    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")
    return word_embedding_final



def my_model(features, labels, mode, params):
    sentence = features['text']
    # Get word embeddings for each token in the sentence

    assert (params["vocab_size"], FLAGS.embedding_size)==params["word_embedding"].shape
    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,shape=[params["vocab_size"], FLAGS.embedding_size],
                                 initializer=tf.constant_initializer(params["word_embedding"], dtype=tf.float32))

    sentence = tf.nn.embedding_lookup(embeddings, sentence)  # shape:(batch, sentence_len, embedding_size)
    # add a channel dim, required by the conv2d and max_pooling2d method
    sentence = tf.expand_dims(sentence, -1)  # shape:(batch, sentence_len/height, embedding_size/width, channels=1)
    # sentence = tf.layers.batch_normalization(sentence, training=(mode == tf.estimator.ModeKeys.TRAIN))
    pooled_outputs = []
    for filter_size in params["filter_sizes"]:

        conv = tf.layers.conv2d(
            sentence,
            filters=FLAGS.num_filters,
            kernel_size=[filter_size, FLAGS.embedding_size],
            strides=(1, 1),
            padding="VALID"
        )#activation=tf.nn.relu
        # conv = tf.layers.batch_normalization(conv, training=(mode == tf.estimator.ModeKeys.TRAIN))
        conv=tf.nn.relu(conv)

        # conv = tf.layers.conv2d(
        #     conv,
        #     filters=FLAGS.num_filters,
        #     kernel_size=[filter_size, 1],
        #     strides=(1, 1),
        #     padding="SAME"
        # )#activation=tf.nn.relu
        # conv = tf.nn.relu(conv)
        # b = tf.get_variable("b-%s" % filter_size, [FLAGS.num_filters])  # ADD 2017-06-09
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            # h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

            conv = tf.layers.dropout(conv, params['dropout_rate'],
                                            training=(mode == tf.estimator.ModeKeys.TRAIN))
        # conv = tf.layers.batch_normalization(conv, training=(mode == tf.estimator.ModeKeys.TRAIN))


        pool = tf.layers.max_pooling2d(
            conv,
            pool_size=[FLAGS.sentence_max_len - filter_size + 1, 1],
            strides=(1, 1),
            padding="VALID")



        pooled_outputs.append(pool)

    h_pool = tf.concat(pooled_outputs, 3)  # shape: (batch, 1, len(filter_size) * embedding_size, 1)
    h_pool_flat = tf.reshape(h_pool, [-1, FLAGS.num_filters * len(
        params["filter_sizes"])])  # shape: (batch, len(filter_size) * embedding_size)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
        # h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

        h_pool_flat = tf.layers.dropout(h_pool_flat, params['dropout_rate'],
                                        training=(mode == tf.estimator.ModeKeys.TRAIN))
    h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(h_pool_flat, FLAGS.num_classes, activation=None)

    # learning_rate = tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(), FLAGS.decay_steps,
    #                                            FLAGS.decay_rate, staircase=True)
    learning_rate=params['learning_rate']
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    def _train_op_fn(loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # def loss_fn_(labels, logits):  # 0.001
    #     l2_lambda = 0.0001
    #     with tf.name_scope("loss"):
    #         # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
    #         # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
    #         losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
    #                                                                 logits=logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
    #         # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
    #         loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
    #         l2_losses = tf.add_n(
    #             [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
    #         loss = loss + l2_losses
    #     return loss

    my_head = tf.contrib.estimator.multi_class_head(n_classes=FLAGS.num_classes)
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=_train_op_fn
    )




def my_head():

    with tf.variable_scope('ctr_model'):
        ctr_logits = build_mode(features, mode, params)
    with tf.variable_scope('cvr_model'):
        cvr_logits = build_mode(features, mode, params)

    ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
    cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
    prop = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': prop,
            'ctr_probabilities': ctr_predictions,
            'cvr_probabilities': cvr_predictions
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    y = labels['cvr']
    cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, prop), name="cvr_loss")
    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['ctr'], logits=ctr_logits),
                             name="ctr_loss")
    loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")

    ctr_accuracy = tf.metrics.accuracy(labels=labels['ctr'],
                                       predictions=tf.to_float(tf.greater_equal(ctr_predictions, 0.5)))
    cvr_accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
    ctr_auc = tf.metrics.auc(labels['ctr'], ctr_predictions)
    cvr_auc = tf.metrics.auc(y, prop)
    metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
    tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
    tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('cvr_auc', cvr_auc[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


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
    params={
        'vocab_size': config["vocab_size"],
        'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate
    }
    word_embedding = assign_pretrained_word_embedding(params)
    params['word_embedding']=word_embedding
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)

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
    input_fn_for_eval = lambda: train_input_fn(path_eval, 0,shuffle=False)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, steps=60, throttle_secs=100)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    print("evalue train set")
    input_fn_for_pred = lambda: train_input_fn(path_train, 0)
    classifier.evaluate(input_fn=input_fn_for_pred,steps=120)

    input_fn_for_eval = lambda: train_input_fn(path_eval, 0)
    # eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, steps=30, throttle_secs=60)
    # tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    print("evalue eval set")
    classifier.evaluate(input_fn=input_fn_for_eval,steps=100)
    print("after train and evaluate")
json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
with open(json_path) as f:
    config = json.load(f)

params = {
    'vocab_size': config["vocab_size"],
    'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
    'learning_rate': FLAGS.learning_rate,
    'dropout_rate': FLAGS.dropout_rate
}

word_embedding = assign_pretrained_word_embedding(params)

def pred_per_file(file_path):

    input_fn_for_pred = lambda: pred_input_fn(file_path, shuffle_buffer_size=0,shuffle=False,repeat=1)
    json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
    with open(json_path) as f:
        config = json.load(f)
    params = {
        'vocab_size': config["vocab_size"],
        'filter_sizes': list(map(int, FLAGS.filter_sizes.split(','))),
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate
    }

    params['word_embedding']=word_embedding
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params=params,
        config=tf.estimator.RunConfig(keep_checkpoint_max=2,model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    eval_spec = classifier.predict(input_fn=input_fn_for_pred)

    count = 0

    result=[]
    label=[]
    prob=[]
    for e in eval_spec:
        count += 1
        result.append((file_path,count,int(list(e.get('classes', ''))[0])))
        label.append(int(list(e.get('classes', ''))[0]))
        prob.append(e.get('probabilities'))
    # for r in result:
    #     print(r)

    return label,prob

def real_one(input_filename):
    label_list=[]
    id_list=[]
    for serialized_example in tf.python_io.tf_record_iterator(input_filename):
        # Get serialized example from file
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        label = example.features.feature["label"]
        features = example.features.feature["_id"]
        label_list.append(label.int64_list.value[0])
        id_list.append(features.int64_list.value[0])
    return label_list,id_list





if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=pred)
