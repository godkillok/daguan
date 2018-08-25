#!/usr/bin/env python
import tensorflow as tf
import os
import re
import numpy as np
import random

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/home/tom/new_data/input_data", "Directory containing the dataset")
flags.DEFINE_string("pad_word", "0", "used for pad sentence")
flags.DEFINE_string("path_vocab", "/home/tom/new_data/input_data/words.txt", "used for word index")
FLAGS = flags.FLAGS

sentence_max_len = 100
pad_word = FLAGS.pad_word

label_class=[]
def feature_auto(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, list):
        try:
            tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        except:
            print(value)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    elif isinstance(value, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    elif isinstance(value, float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def parse_line_dict(record):
    fields = record.split(",")
    if len(fields) < 2:
        raise ValueError("invalid record %s" % record)
    text = fields[1].strip().lower()
    tokens = text.split(' ')
      # type: int
    if len(fields) == 2:
        label=0
    else:
        label = int(fields[2]) - 1
    label_class.append(label)
    return [tokens, label]


def per_thouds_lines_dict(result_lines, vocab, path_text, count,flag_name):
    tf_lines = []
    for rl in result_lines:
        text = [vocab.get(r) for r in rl[0] if vocab.get(r, '-99')!='-99' ]
        label = rl[1]
        n = len(text)
        if n > sentence_max_len:
            text = text[:sentence_max_len]
        if n < sentence_max_len:
            text += [0] * (sentence_max_len - n)
        tf_lines.append([text, label])

    write_tfrecords(tf_lines, path_text, count,flag_name)


def generate_tf_dic(path_text):
    with open(FLAGS.path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {l.strip(): i for i, l in enumerate(lines)}

    result_lines = []
    count = 0
    with open(path_text, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            field=line.split(',')
            count += 1
            try:
                int(field[0])
            except:
                continue
            result_lines.append(parse_line_dict(line))
            if count % 20000 == 0 or count == len(lines) - 1:
                if len(field)>2:
                    if count<92277:
                        flag_name='train'
                    else:
                        flag_name = 'eval'
                else:
                    flag_name = 'pred'
                per_thouds_lines_dict(result_lines, vocab, path_text, count,flag_name)
                result_lines = []


def write_tfrecords(tf_lines, path_text, count,flag_name):
    (root_path, output_filename) = os.path.split(path_text)
    output_filename = output_filename.split('.')[0]
    output_file = output_filename + '_' + str(count) + '_' + str(0) +'_' + flag_name+ '.tfrecords'

    print("Start to convert {} to {}".format(len(tf_lines), os.path.join(root_path, output_file)))

    writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
    random.shuffle(tf_lines)
    num = 0
    for data in tf_lines:
        text = data[0]
        label = data[1]
        example = tf.train.Example(features=tf.train.Features(feature={
            'text': feature_auto(list(text)),
            'label': feature_auto(int(label))
        }))

        writer.write(example.SerializeToString())
        num += 1
        if num % 1000 == 0:
            output_file = output_filename + '_' + str(count) + '_' + str(num)+'_' + flag_name + '.tfrecords'
            writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
            print("Start convert to {}".format(output_file))


def main():
    s3_input = FLAGS.data_dir
    for root, dirs, files in os.walk(s3_input):
        for file in files:
            if file.endswith(".csv"):
                print('start to process file {}'.format(file))
                generate_tf_dic(os.path.join(root, file))
    print(set(label_class))


if __name__ == "__main__":
    main()
