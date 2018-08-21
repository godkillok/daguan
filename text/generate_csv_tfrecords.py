#!/usr/bin/env python
import tensorflow as tf
import os
import re
import numpy as np
import random

flags = tf.app.flags
flags.DEFINE_string("data_dir", "./dbpedia_csv", "Directory containing the dataset")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
flags.DEFINE_string("path_vocab", "/dbpedia_csv/words.txt", "used for word index")
FLAGS = flags.FLAGS


sentence_max_len = 100
pad_word = FLAGS.pad_word


def feature_auto(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value,list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    elif isinstance(value, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    elif isinstance(value, float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def parse_line(line, vocab):
    def get_content(record):
        fields = record.decode().split(",")
        if len(fields) < 3:
            raise ValueError("invalid record %s" % record)
        text = re.sub(r"[^A-Za-z0-9\'\`]", " ", fields[2])
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\`", "\'", text)
        text = text.strip().lower()
        tokens = text.split()
        tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0]
        n = len(tokens)  # type: int
        if n > sentence_max_len:
            tokens = tokens[:sentence_max_len]
        if n < sentence_max_len:
            tokens += [pad_word] * (sentence_max_len - n)
        return [tokens, np.int32(fields[0])]

    result = tf.py_func(get_content, [line], [tf.string, tf.int32])
    result[0].set_shape([sentence_max_len])
    result[1].set_shape([])
    # Lookup tokens to return their ids
    ids = vocab.lookup(result[0])

    sess=tf.Session()
    with sess.as_default():
        tf.tables_initializer().run()
        ge=ids.eval()
        label=(result[1] - 1).eval()
    return {"sentence": ge, "label":label}


def generate_tfrecords(path_text):
    path_vocab = os.path.join(FLAGS.path_vocab)
    vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=1)
    tf_lines=[]
    with open(path_text,'r',encoding='utf8') as f:
        lines=f.readlines()
        for line in lines:
            tf_lines.append(parse_line(line,vocab))
    write_tfrecords(tf_lines,path_text)

def write_tfrecords(tf_lines, path_text):

    (root_path,output_filename)=os.path.split(path_text)
    output_filename=output_filename.split('.')[0]
    output_file=output_filename+'.tfrecords'

    print("Start to convert {} to {}".format(len(tf_lines), os.path.join(root_path,output_file)))

    writer = tf.python_io.TFRecordWriter(os.path.join(root_path,output_file))
    count = 0
    random.shuffle(tf_lines)
    for data in tf_lines:
        text = data.get('sentence')
        label = data.get('label')
        example = tf.train.Example(features=tf.train.Features(feature={
            'text': feature_auto(list(text)),
            'label': feature_auto(int(label))
        }))

        writer.write(example.SerializeToString())
        if count%1000==0:
            output_file=output_filename+'_'+str(count)+'.tfrecords'
            writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))

            print("Start convert to {}".format(output_file))


def main():
    s3_input=FLAGS.data_dir
    for root, dirs, files in os.walk(s3_input):
        for file in files:
            if file.endswith(".csv"):
                generate_tfrecords(os.path.join(root, file))


if __name__ == "__main__":
    main()

