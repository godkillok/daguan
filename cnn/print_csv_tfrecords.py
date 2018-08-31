#!/usr/bin/env python

import tensorflow as tf
import os


def print_tfrecords(input_filename):
  max_print_number = 100
  current_print_number = 0

  for serialized_example in tf.python_io.tf_record_iterator(input_filename):
    # Get serialized example from file
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    label = example.features.feature["label"]
    features = example.features.feature["_id"]
    print("Number: {}, label: {}, features:".format(current_print_number,
                                                    features.int64_list.value[0]))

    # Return when reaching max print number
    current_print_number += 1
    # print(current_print_number)
    # if current_print_number > max_print_number:
    #   exit()[13]


def main():
  current_path = os.getcwd()
  tfrecords_file_name = "test_100000_30000_pred.tfrecords"
  # input_filename = os.path.join(current_path, tfrecords_file_name)
  input_filename='/home/tom/new_data/input_data/'+tfrecords_file_name
  print_tfrecords(input_filename)


if __name__ == "__main__":
  main()
