#!/usr/bin/env python
import tensorflow as tf
import os
import re
import numpy as np
import random

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/home/tom/new_data/daguan/text/dbpedia_csv", "Directory containing the dataset")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
flags.DEFINE_string("path_vocab", "/home/tom/new_data/daguan/text/dbpedia_csv/words.txt", "used for word index")
FLAGS = flags.FLAGS

sentence_max_len = 100
pad_word = FLAGS.pad_word


def feature_auto(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, list):
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

    return result

def per_thouds_lines(result_lines,vocab,path_text,count):

    text=[r[0] for r in result_lines]
    labels=[r[1]-1 for r in result_lines]
    text_tensor=tf.reshape(text, [len(text), -1])
    labels_tensor = tf.reshape(labels, [len(labels), -1])

    with tf.Session() as  sess:
        tf.tables_initializer().run()
        ids = vocab.lookup(text_tensor)
        ggtext=ids.eval()
        ggl=labels_tensor.eval()
        sess.close()
        # for t in text:
        #     ids = vocab.lookup(t)
        #     ge.append(ids.eval())
        # for l in labels:
        #     label.append(l.eval())
    tf_lines=[]
    print(len(ggtext))
    for (g,l) in zip(ggtext,ggl):
        tf_lines.append([g,l])
    write_tfrecords(tf_lines, path_text,count)

def generate_tfrecords(path_text):
    path_vocab = os.path.join(FLAGS.path_vocab)
    vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=1)
    result_lines = []
    count=0
    with open(path_text, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            count+=1
            result_lines.append(parse_line(line, vocab))
            if count%10000==0 or count==len(lines)-1:
                per_thouds_lines(result_lines,vocab, path_text, count)
                result_lines=[]


def write_tfrecords(tf_lines, path_text,count):
    (root_path, output_filename) = os.path.split(path_text)
    output_filename = output_filename.split('.')[0]
    output_file = output_filename + '_' + str(count)+ '_'+  str(0) + '.tfrecords'

    print("Start to convert {} to {}".format(len(tf_lines), os.path.join(root_path, output_file)))

    writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
    random.shuffle(tf_lines)
    num=0
    for data in tf_lines:
        text = data[0]
        label = data[1]
        example = tf.train.Example(features=tf.train.Features(feature={
            'text': feature_auto(list(text)),
            'label': feature_auto(int(label))
        }))

        writer.write(example.SerializeToString())
        num+=1
        if num % 1000 == 0:
            output_file = output_filename + '_' + str(count) + '_' + str(num)+ '.tfrecords'
            writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
            print("Start convert to {}".format(output_file))


def main():
    s3_input = FLAGS.data_dir
    for root, dirs, files in os.walk(s3_input):
        for file in files:
            if file.endswith("t_shuf.csv"):
                print('start to process file {}'.format(file))
                generate_tfrecords(os.path.join(root, file))


if __name__ == "__main__":
    main()
