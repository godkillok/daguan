#!/usr/bin/env python

import tensorflow as tf
import os


def make_example(key):

    example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
            "length":tf.train.Feature(int64_list=tf.train.Int64List(value=[len(key)]))
        }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
            "index":tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[key[i]])) for i in range(len(key))])
            }
        )
    )
    return example.SerializeToString()


filename="tmp.tfrecords"
if os.path.exists(filename):
    os.remove(filename)
writer = tf.python_io.TFRecordWriter(filename)



def generate_tfrecords(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))
  writer = tf.python_io.TFRecordWriter(output_filename)

  index = 0
  for line in open(input_filename, "r"):
    index += 1

    data = line.split(",")
    label = float(data[14])
    features = [float(i) for i in data[1:14]]

    example = tf.train.Example(features=tf.train.Features(
        feature={
            "label":
            tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
            "features":
            tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        }))
    writer.write(example.SerializeToString())

  writer.close()
  print(
      "Successfully convert {} to {}".format(input_filename, output_filename))


def main():
  current_path = os.getcwd()
  for filename in os.listdir(current_path):
    if filename.startswith("") and filename.endswith(".csv"):
      generate_tfrecords(filename, filename + ".tfrecords")


if __name__ == "__main__":
  main()