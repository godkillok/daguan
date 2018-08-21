#!/usr/bin/env python

import tensorflow as tf
import os
import csv
import re
import numpy as np

# 将数据转化成对应的属性
sentence_max_len = 100
pad_word = '<pad>'


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


def gete():
    path_text = os.path.join('./text.csv')
    path_vocab = os.path.join('./words.txt')
    vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=1)
    tf_lines=[]
    with open(path_text,'r',encoding='utf8') as f:
        lines=f.readlines()
        for line in lines:
            tf_lines.append(parse_line(line,vocab))
    output_filename='./1.tfrecords'
    generate_tfrecords(tf_lines,output_filename)

def generate_tfrecords(tf_lines, output_filename):
    print("Start to convert {} to {}".format(len(tf_lines), output_filename))
    writer = tf.python_io.TFRecordWriter(output_filename)
    cout = 0

    for data in tf_lines:
        text = data.get('sentence')
        label = data.get('label')
        example = tf.train.Example(features=tf.train.Features(feature={
            'text': feature_auto(list(text)),
            'label': feature_auto(int(label))
        }))
        if cout < 8:
            writer.write(example.SerializeToString())


    print("Successfully convert {} to {}".format(len(tf_lines),
                                                 output_filename))


def main():
    current_path = os.getcwd()
    for filename in os.listdir(current_path):
        if filename.startswith("") and filename.endswith(".csv"):
            generate_tfrecords(filename, filename + ".tfrecords")


if __name__ == "__main__":
    # main()
    gete()
