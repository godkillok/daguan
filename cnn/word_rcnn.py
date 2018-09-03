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
path='C:/Users/TangGuoping/Downloads'
flags.DEFINE_string("model_dir", "./model_dir1", "Base directory for the model.")
flags.DEFINE_string("train_file_pattern", "{}/*train.tfrecords".format(path), "train file pattern")
flags.DEFINE_string("eval_file_pattern", "{}/*eval.tfrecords".format(path), "evalue file pattern")
flags.DEFINE_float("dropout_rate", 0.8, "Drop out rate")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
flags.DEFINE_float("decay_rate", 0.7, "Learning rate")
flags.DEFINE_integer("embedding_size", 100, "embedding size")
flags.DEFINE_integer("num_filters", 100, "number of filters")
flags.DEFINE_integer("num_classes", 19, "number of classes")
flags.DEFINE_integer("num_parallel_readers", 4, "number of classes")
flags.DEFINE_integer("shuffle_buffer_size", 30000, "dataset shuffle buffer size")
flags.DEFINE_integer("sentence_max_len", 250, "max length of sentences")
flags.DEFINE_integer("batch_size", 128, "number of instances in a batch")
flags.DEFINE_integer("save_checkpoints_steps", 500, "Save checkpoints every this many steps")
flags.DEFINE_integer("train_steps", 60000,
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
flags.DEFINE_string("fast_text", "/home/tom/new_data/super.bin", "used for word index")
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
    # sentence = tf.expand_dims(sentence, -1)  # shape:(batch, sentence_len/height, embedding_size/width, channels=1)
    # sentence = tf.layers.batch_normalization(sentence, training=(mode == tf.estimator.ModeKeys.TRAIN))
    pooled_outputs = []

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
    pooled_outputs=[]
    pooled_outputs.append(output_rnn_last)
    pooled_outputs.append(sentence)
    h_pool = tf.concat(pooled_outputs, 3)  # shape: (batch, 1, len(filter_size) * embedding_size, 1)









    h_pool_flat = tf.reshape(h_pool, [-1, FLAGS.num_filters * len(
        params["filter_sizes"])])  # shape: (batch, len(filter_size) * embedding_size)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
        # h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

        h_pool_flat = tf.layers.dropout(h_pool_flat, params['dropout_rate'],
                                        training=(mode == tf.estimator.ModeKeys.TRAIN))
    h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(h_pool_flat, FLAGS.num_classes, activation=None)

    logits = tf.layers.dense(output_rnn_last, FLAGS.num_classes, activation=None)

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
    # word_embedding = assign_pretrained_word_embedding(params)
    vocab_size = params["vocab_size"]
    word_embedding = np.random((vocab_size, FLAGS.embedding_size))
    params['word_embedding'] = word_embedding
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


def pred(unused_argv):
    import re
    from collections import Counter
    acc_list=[]
    from collections import defaultdict
    dic = defaultdict(list)
    local_path, pattern = os.path.split(FLAGS.pred_file_pattern)
    for root, dirs, files in os.walk(local_path):
        for file in files:

            regu_cont = re.compile(r'.{}'.format(pattern), re.I)
            if regu_cont.match(file):
                right = 0
                wrong = 0
                print(file)
                file_path = os.path.join(root, file)
                # file_path='/home/tom/new_data/input_data/train_shuf_100000_11000_eval.tfrecords'
                label=pred_per_file(file_path)
                label_list,id_list=real_one(file_path)


                for pl,_lb,_id in zip(label,label_list,id_list):
                    dic[_id].append(pl)
    base_line={}
    with open('/home/tom/Desktop/baseline.csv','r') as f:
        lines=f.readlines()
        for l in lines:
            _id,lb=l.split(',')
            base_line[_id]=lb
    pred_res=[]
    for (k, v_list) in dic.items():
        sv = Counter(v_list).most_common(2)
        print('result:'+str(k)+'\t'+str(len(v_list))+'\t',sv)
    # with open('/home/tom/Desktop/wcnn.txt','w',encoding='utf8') as f:
    #     for (k,v_list) in dic.items():
    #         sv=Counter(v_list).most_common(2)
    #         f.writelines('{},{}\n'.format(k+'\t'+str(len(v_list))+'\t',sv))
        # if len(sv)>1:
        #     if sv[0][1]>sv[1][1]:
        #         pred_res.append((k,sv[0][0]))
        #     else:
        #         v_list.append()




def pred_eval(unused_argv):
    import re
    acc_list=[]
    local_path, pattern = os.path.split(FLAGS.pred_file_pattern)
    for root, dirs, files in os.walk(local_path):
        for file in files:

            regu_cont = re.compile(r'.{}'.format(pattern), re.I)
            if regu_cont.match(file):
                right = 0
                wrong = 0
                print(file)
                file_path = os.path.join(root, file)
                # file_path='/home/tom/new_data/input_data/train_shuf_100000_11000_eval.tfrecords'
                label,acc=pred_per_file(file_path)
                label_list,id_list=real_one(file_path)
                from collections import defaultdict
                dic=defaultdict(list)
                for pl,rl,_id in zip(label,label_list,id_list):
                    dic[_id].append(pl==rl)
                    if pl==rl:
                        right+=1
                    else:
                        wrong+=1
                # print(dic)
                print(right,wrong,acc)
                right = 0
                wrong = 0
                for v_list in dic.values():
                    tn = 0
                    fn = 0
                    for v in v_list:
                        if v:
                            tn += 1
                        else:
                            fn += 1
                        if tn > fn:
                            right += 1
                        elif tn < fn:
                            wrong += 1
                        elif tn == fn:

                            if v_list[0]:
                                right += 1
                            else :
                                wrong += 1
                print(right, wrong, right/(right+wrong))
                print('gg')
                acc_list.append((acc,right/(right+wrong)))
    for a in acc_list:
        print(a)



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
